from dataclasses import dataclass
from typing import Dict, List, Any
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]
    
    def rename_with_best_practices(self) -> None:
        """Rename the columns with best practices (e.g., snake_case very descriptive names)."""
        if self.data:
            renamed_data = []
            for entry in self.data:
                renamed_entry = {}
                for key in entry.keys():
                    original_key = key
                    
                    # Convert the key to snake_case
                    new_key = re.sub(r'(?<!^)(?=[A-Z])', '_', key)  # Underscore before capital letters
                    new_key = re.sub(r'[^a-zA-Z0-9]', '_', new_key)  # Replace non-alphanumerics with underscores
                    new_key = re.sub(r'__+', '_', new_key)  # Replace multiple underscores with a single one
                    new_key = new_key.lower()  # Convert to lowercase
                    
                    # If new_key starts with a digit, modify it appropriately
                    if new_key and new_key[0].isdigit():
                        # Remove leading digit entirely or describe its significance
                        new_key = 'num_' + new_key  # Instead of keeping the number, we add significance; adjust as needed.

                    # Strip any leading/trailing underscores (in case transformation added them)
                    new_key = new_key.strip('_')

                    renamed_entry[new_key] = entry[key]
                    
                renamed_data.append(renamed_entry)
            self.data = renamed_data
        return self.data

    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace 'NA' with None in all values of the dictionary and convert numeric strings to floats."""
        for entry in self.data:
            for key, value in entry.items():
                if value == "NA":
                    entry[key] = None  # Replace "NA" with None
                elif isinstance(value, str) and value.isnumeric():
                    entry[key] = float(value)  # Convert numeric strings to float
        return self.data  # Return modified data

