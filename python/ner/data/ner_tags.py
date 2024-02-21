from enum import Enum


class NERType(Enum):
    VOID = 'VOID'
    ART = 'ART'
    BUILDING = 'BUILDING'
    EVENT = 'EVENT'
    LOCATION = 'LOCATION'
    ORGANIZATION = 'ORGANIZATION'
    OTHER = 'OTHER'
    PERSON = 'PERSON'
    PRODUCT = 'PRODUCT'
