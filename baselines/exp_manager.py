from dataclasses import dataclass

@dataclass
class experience_manager:
    script: str
    headless: bool
    use_sde: bool
    

    def distance_to_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
