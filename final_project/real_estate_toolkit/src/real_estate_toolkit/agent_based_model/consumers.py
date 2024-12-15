from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
from real_estate_toolkit.agent_based_model.common import Segment
from real_estate_toolkit.agent_based_model.houses import House  # Importing from agent_based_model folder
from real_estate_toolkit.agent_based_model.market import HousingMarket # Import the HousingMarket class


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None  # Default to None, as no house is purchased initially
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    
    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time.
        """
        annual_saving = self.annual_income * self.saving_rate
        for _ in range(years):
            # Add the annual savings first
            self.savings += annual_saving
            # Apply interest to the total savings
            self.savings *= (1 + self.interest_rate)
        # Round the savings to 2 decimal places
        self.savings = round(self.savings, 2)

        print(f"Consumer {self.id} has accumulated {self.savings} savings.")

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house.

        """
        # Step 1: Define house filtering criteria
        max_price = self.savings  # Consumer can only afford houses within their savings

        # Filter houses based on segment preferences
        if self.segment == Segment.FANCY:
            # Fancy segment: prioritize new construction and highest house score
            suitable_houses = [
                house for house in housing_market.houses
                if house.is_new_construction() and 
                   (house.quality_score is not None and house.quality_score.value == max(house.quality_score.value for house in housing_market.houses if house.quality_score is not None))
            ]
        elif self.segment == Segment.OPTIMIZER:
            # Optimizer segment: prioritize houses with price per square foot < monthly salary
            monthly_salary = self.annual_income / 12
            suitable_houses = [
                house for house in housing_market.houses
                if house.calculate_price_per_square_foot() < monthly_salary
            ]
        elif self.segment == Segment.AVERAGE:
            # Average segment: prioritize houses below the average market price
            average_price = sum(house.price for house in housing_market.houses) / len(housing_market.houses) if housing_market.houses else 0
            suitable_houses = [
                house for house in housing_market.houses
                if house.price < average_price
            ]
        else:
            print("Segment not recognized.")
            return

        # Step 2: Further filter houses within the consumer's budget
        affordable_houses = [
            house for house in suitable_houses
            if house.price <= max_price
        ]

        # Step 3: Match house to family size needs
        family_size = 1 + self.children_number  # Includes the consumer
        suitable_for_family = [
            house for house in affordable_houses
            if house.bedrooms >= family_size
        ]

        # Step 4: Select the best house (e.g., the cheapest house that meets all criteria)
        if suitable_for_family:
            best_house = min(suitable_for_family, key=lambda h: h.price)
            self.house = best_house
            self.savings -= best_house.price  # Deduct the price of the house from savings
            print(f"Consumer {self.id} purchased house {best_house.id} at {best_house.price}.")
        else:
            print(f"Consumer {self.id} could not find a suitable house to purchase.")