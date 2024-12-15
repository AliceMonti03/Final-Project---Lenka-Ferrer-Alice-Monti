from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, choice
from typing import Optional, List, Dict, Any
from real_estate_toolkit.agent_based_model.houses import House  
from real_estate_toolkit.agent_based_model.market import HousingMarket 
from real_estate_toolkit.agent_based_model.consumers import Consumer, Segment  

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float

@dataclass
class ChildrenRange:
    minimum: float = 0
    maximum: float = 5

@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    
    def create_housing_market(self):
        """
        Initialize market with houses.
        
        """
        # Step 1: Validate the input data with actual field names from the CSV
        required_fields = {"id", "sale_price", "bedroom_abv_gr", "year_built", "lot_area"}
        for house_data in self.housing_market_data:
            if not required_fields.issubset(house_data.keys()):
                raise ValueError(f"Missing required fields in house data: {house_data}")

        # Step 2: Create House objects
        houses = []
        for house_data in self.housing_market_data:
            house = House(
                id=house_data["id"],
                price=house_data["sale_price"],
                area=house_data["lot_area"],  
                bedrooms=house_data["bedroom_abv_gr"],
                year_built=house_data["year_built"],
                quality_score=None  # Can be calculated later if needed
            )
            houses.append(house)

        # Step 3: Assign the housing market
        self.housing_market = HousingMarket(houses=houses)
            
    def create_consumers(self) -> None:

        consumers = []

        for i in range(self.consumers_number):
            # Step 1: Generate annual income using a truncated normal distribution
            while True:
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
                if self.annual_income.minimum <= income <= self.annual_income.maximum:
                    break

            # Step 2: Generate the number of children using a random integer
            children_number = randint(self.children_range.minimum, self.children_range.maximum)

            # Step 3: Randomly assign a segment
            segment = choice(list(Segment))

            # Step 4: Create a Consumer object
            consumer = Consumer(
                id=i + 1,  # Assign a unique ID to each consumer
                annual_income=income,
                children_number=children_number,
                segment=segment,
                house=None,  # No house assigned initially
                savings=0.0,  # Default initial savings
                saving_rate=self.saving_rate,  # Assign the simulation's saving rate
                interest_rate=self.interest_rate  # Assign the simulation's interest rate
            )

            # Add the consumer to the list
            consumers.append(consumer)

        # Assign the list of consumers to a class property
        self.consumers = consumers
    
 
    def compute_consumers_savings(self) -> None:
       
        if not self.consumers:
            raise ValueError("Consumers have not been created. Please run create_consumers() first.")

        for consumer in self.consumers:
            # Annual savings contribution
            annual_contribution = consumer.annual_income * consumer.saving_rate

            # Compound interest formula for savings
            accumulated_savings = (
                consumer.savings * (1 + self.interest_rate) ** self.years +
                annual_contribution * (((1 + self.interest_rate) ** self.years - 1) / self.interest_rate)
            )

            # Update the consumer's savings
            consumer.savings = round(accumulated_savings, 2)

        print("Consumer savings have been computed.")

    def clean_the_market(self) -> None:

        if not self.consumers or not self.housing_market:
            raise ValueError("Consumers or housing market are not initialized. Please ensure both are created.")

        # Step 1: Sort consumers based on the cleaning mechanism
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            sorted_consumers = sorted(self.consumers, key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            sorted_consumers = sorted(self.consumers, key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            from random import shuffle
            sorted_consumers = self.consumers[:]
            shuffle(sorted_consumers)
        else:
            raise ValueError("Invalid cleaning market mechanism.")

        # Step 2: Execute market transactions
        successful_purchases = 0
        for consumer in sorted_consumers:
            if consumer.house is None:  # Only attempt to buy if the consumer hasn't purchased a house
                consumer.buy_a_house(self.housing_market)
                if consumer.house:
                    successful_purchases += 1

        # Step 3: Log the results
        print(f"Market cleaning completed. Total successful purchases: {successful_purchases}.")
        remaining_houses = len(self.housing_market.houses)
        print(f"Houses remaining in the market: {remaining_houses}.")
    
    def compute_owners_population_rate(self) -> float:

        if not self.consumers:
            raise ValueError("Consumers have not been initialized. Please run create_consumers() first.")

        # Count consumers who have successfully purchased a house
        owners_count = sum(1 for consumer in self.consumers if consumer.house is not None)

        # Calculate the ownership rate
        ownership_rate = owners_count / self.consumers_number

        print(f"Ownership rate: {ownership_rate:.2%}")
        return ownership_rate
    
    def compute_houses_availability_rate(self) -> float:
    
        if not self.housing_market:
            raise ValueError("Housing market has not been initialized. Please run create_housing_market() first.")

        # Calculate the total number of houses
        total_houses = len(self.housing_market_data)

        # Calculate the number of available houses
        available_houses = len(self.housing_market.houses)

        # Compute the availability rate
        availability_rate = available_houses / total_houses

        print(f"Houses availability rate: {availability_rate:.2%}")
        return availability_rate