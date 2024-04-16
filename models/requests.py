from pydantic import BaseModel, validator

class Request(BaseModel):
    numberofdays: int = 120

    @validator('numberofdays')
    def check_non_zero(cls, v):
        if v == 0:
            raise ValueError('numberofdays must be greater than zero')
        if v >= 121:
            raise ValueError('numberofdays must be lower than 121')
        return v
