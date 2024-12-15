from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True
    
    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        
        Implementation tips:
        - Divide price by area
        - Round to 2 decimal places
        - Handle edge cases (e.g., area = 0)
        """
        # Handle edge case where area is zero to avoid division by zero.
        if self.area <= 0:
            raise ValueError("Area must be greater than zero to calculate price per square foot.")

        # Calculate the price per square foot
        price_per_sqft = self.price / self.area

        # Return the price per square foot rounded to 2 decimal places
        return round(price_per_sqft, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).
        """
        if not isinstance(self.year_built, (int, float)):
            raise ValueError("Year built must be a valid integer or float.")
        
        return (current_year - self.year_built) < 5
    
    def get_quality_score(self, current_year: int = 2024) -> None:
        """
        Generate a quality score based on house attributes.
        """
        # Calculate the age based on the current year
        age = current_year - self.year_built
        
        # Basic scoring logic based on attributes
        if self.area >= 2000 and self.bedrooms >= 3:
            score = QualityScore.EXCELLENT
        elif self.area >= 1500 and self.bedrooms >= 2:
            score = QualityScore.GOOD
        elif self.bedrooms == 1 or age > 30:
            score = QualityScore.FAIR
        else:
            score = QualityScore.POOR
        
        # Assign the calculated score to the quality_score attribute
        self.quality_score = score

    def sell_house(self) -> None:
        """
        Mark the house as sold by setting its availability status to False.
        """
        # Directly mark the house as sold
        self.available = False