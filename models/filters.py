from typing import List, Optional
from pydantic import BaseModel, Field, validator


class Offenses(BaseModel):
    """
    This class contains attriburtes around offenses like current or prior offenses, 
    name of offenses etc
    """
    offense_number: str =  Field(..., description="The offense number from the California criminal code (penal code, vehicular code, other). This is of the type PC123, VC666, PC456, etc.")
    offense_number_column: str =  Field(..., description="The name of the field or variable in the context that relates to offense numbers from the California criminal code (penal code, vehicular code, other)")

    offense_number_tables: str = Field(..., description="Specifies the tables to search for the offense number. These are of the form 'Table' followed by labels like 'A', 'B', 'C' which denote the list of possible values to search")
    offense_number_type: str = Field(..., description="Specifies the kind or category of offenses to search for or query. Categorize offenses type as 'controlling', 'current' or 'prior' or 'unknown'")
    offense_number_operator: str = Field(..., description="Specifies how to search or compare offense numbers. This could be looking for an exact match or whether the offense number is included or excluded in the tables mentioned. To denote conditional operators for text or string value matching in a table use 'include', 'exclude'. For string matching against another string use the 'exact' operator")
    
    offense_description: str = Field(..., description="Natural or human language description of the offense number. This could be robbery, petty theft, assault, drug possession, etc.")
    offense_description_column: str = Field(..., description="The name of the field or variable in the context that relates to the description of the offense number.")
    
    offense_subquery: str = Field(..., description="Sub part of user query that contains infomration about offenses to support other attriabutes of this class.")



class Demographics(BaseModel):
    """
    This class contains attributes around demographics like age of sentenced individuals, 
    number of years they have been senetenced for, controlling offenses number,penal code etc. 
    
    The values of these should always be valid values mentioned in the user query, in case of no value the field should be left to default values.
    """

    individual_age: int = Field(..., description="A numerical value that represents an individual's age")
    individual_age_column: str = Field(..., description="The name of the column or field in the context that relates to a population's age")
    individual_age_operator: str = Field(..., description="Logical operator for the age of the population")

    individual_location: str = Field(..., description="A text value that represents the institution or facility where the individual is imprisoned or incarcerated. This could be San Quentin, CA Institution for Men etc.")
    individual_location_column: str = Field(..., description="The name of the column or field in the context that relates to the location where the individual is imprisoned")
    
    sentenced_by: str = Field(..., description="A text value that represents the county or geolocation where the individual was sentenced or where the offense occured. This could be 'Santa Clara', 'San Francisco', 'Out of country', 'Another state', etc.")
    sentenced_by_column: str = Field(..., description="The name of the column or field in the context that relates to the location where the individual was sentenced or where the individual committed their offense")
    
    individual_ethnicity: str = Field(..., description="This should be values like (White, black, Asian, Hispanic, Pacific Islander, etc.) representing Ethnicity or Race of the incarcerated person ")
    individual_ethnicity_column: str = Field(..., description="The name of the column or field in the context that relates to race or ethnicity of a population")
    
    sentenced_age: int = Field(..., description="A numerical value that represents the age at which an individual was sentenced")
    sentenced_age_column: str = Field(..., description="The name of the column or field in the context that relates to the age at which an individual was sentenced")
    sentenced_age_operator: str = Field(..., description="Logical operator for the age of sentenced individuals")
    
    sentenced_years: int = Field(..., description="A numerical value for the number of years an individual was sentenced for")
    sentenced_years_column: str = Field(..., description="The name of the column or field in the context that relates to the length of an individual's sentence or the number of years they were sentenced for")
    sentenced_years_operator: str = Field(..., description="Condional operator for number of years sentenced mentioned. Do not provide any operator in case no sentenced years in user query. To denote conditional operators for numerical values use >, <, =, !=, >=, <=  ")

    served_years: int = Field(..., description="A numerical value for the number of years an individual has served time for or been imprisoned for. This is the number of years of the total sentence length that they have completed.")
    served_years_column: str = Field(..., description="The name of the column or field in the context that relates to the number of years they have served time for or been imprisoned for")
    served_years_operator: str = Field(..., description="Condional operator for number of years served mentioned. To denote conditional operators for numerical values use >, <, =, !=, >=, <=  ")
    
    demographic_subquery: str = Field(..., description="Sub part of the user query that contains information about demographics to support other attriabutes of this class.")



class Conditions(BaseModel):
    """Information about logical conditions about individuals sentenced for offenses."""

    demographics: Optional[Demographics] = Field(..., description="Provides details around demographics as mentioned in the user query")
    offenses: Optional[Offenses] = Field(..., description="Provides details around offenses as mentioned in the user query")
    logical_operations: Optional[List[str]] = Field(..., description="Provide sequence of logical operations between each of the conditions specified by the user. These could be 'AND', 'OR', 'unknown'")



class Filters(BaseModel):
    """Identifying information about logical expressions provided in a text."""

    eligibility_filters: List[Conditions]



class DeconstructedUserQuery(BaseModel):
    """Identifying information about logical expressions provided in a text."""

    deconstructed_query: str = Field(..., description="Deconstructed statement from the user query that may represent a sub query.")
    deconstructed_query_type: str = Field(..., description="Depending upon the content of the statement in the sub query, it can be classified as either of these two types 'demographics' or 'offenses'.")
    deconstructed_query_id: str = Field(..., description="This is a random sequence number allocated to the user sub query based on previous statements in the main query.")



class DeconstructedUserQueries(BaseModel):
    """Identifying mutliple sub queries within user's main query in order to figure out the right logical expressions to 
        represent the various sub queries and the possible logical operations between them.
    """

    deconstructed_queries: List[DeconstructedUserQuery]
