from typing import List, Dict, Optional
from real_estate_toolkit.agent_based_model.houses import House  # Importing from agent_based_model folder
from real_estate_toolkit.agent_based_model.common import Segment
from statistics import mean
 

class HousingMarket:
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """Retrieve specific house by ID."""
        # Iterate through the list of houses to find the one with the matching ID
        for house in self.houses:
            if house.id == house_id:
                # Print the found house's info, excluding address
                print(f"House found: ID={house.id}, Price={house.price}, Area={house.area}, Bedrooms={house.bedrooms}")
                return house  # Return the house if the ID matches
        
        # If no house is found, print a message and return None
        print(f"No house found with ID={house_id}.")
        return None

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        
            # Filter the houses by bedrooms if specified
            if bedrooms is not None:
                filtered_houses = [house.price for house in self.houses if house.bedrooms == bedrooms]
            else:
                filtered_houses = [house.price for house in self.houses]

            # Handle the case where no houses are found
            if not filtered_houses:
                print("No houses match the given criteria.")
                return 0.0

            # Calculate the average price
            avg_price = mean(filtered_houses)
            print(f"Average price calculated: {avg_price:.2f}")
            return avg_price
        
    def get_houses_that_meet_requirements(self, max_price: float, segment: Segment) -> Optional[List[House]]:
        """
        Filter houses based on buyer requirements.
        
        """

        # Calculate average price for the AVERAGE segment filter logic
        average_price = sum(house.price for house in self.houses) / len(self.houses) if self.houses else 0

        # Define static monthly salary thresholds for the segments
        salary_thresholds = {
            Segment.FANCY: 5000,   # Example static threshold for FANCY
            Segment.OPTIMIZER: 3000,  # Example static threshold for OPTIMIZER
            Segment.AVERAGE: 2000   # Example static threshold for AVERAGE
        }

        # Get the salary threshold based on the segment
        monthly_salary = salary_thresholds.get(segment, 0)  # Default to 0 if segment is not found

        # Filter the houses based on the provided maximum price and consumer segment criteria
        # Filter the houses based on the provided maximum price and consumer segment criteria
        matching_houses = []
        for house in self.houses:
            if house.price <= max_price:  # Check price criteria
                if segment == Segment.FANCY:
                    # Criteria for a FANCY house: e.g., new construction
                    if house.is_new_construction():  # Assuming house.has_new_construction() is defined
                        matching_houses.append(house)

                elif segment == Segment.OPTIMIZER:
                    # Criteria for an OPTIMIZER house: e.g., price per square foot
                    if house.calculate_price_per_square_foot() < monthly_salary:
                        matching_houses.append(house)

                elif segment == Segment.AVERAGE:
                    # Criteria for an AVERAGE house: e.g., below average price
                    if house.price < average_price:
                        matching_houses.append(house)

        # Handle the case where no houses match the criteria
        if not matching_houses:
            print(f"No houses found matching the criteria: max_price={max_price}, segment={segment.name}.")
            return None

        # Return the list of matching houses
        print(f"{len(matching_houses)} houses found matching the criteria: max_price={max_price}, segment={segment.name}.")
        return matching_houses