from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import re
from datetime import datetime
from rag.vectorstore import retriever

class Weather(BaseModel):
    location: str 

class ExtractBookingInput(BaseModel):
    message: str

class AvailabilityInput(BaseModel):
    service: str
    preferred_datetime: str

class CreateBookingInput(BaseModel):
    service: str
    preferred_datetime: str
    client_name: str
    client_phone: str

class RAGInput(BaseModel):
    query: str = Field(description="Question to search in knowledge base")
    
@tool(name_or_callable="weather_tool"
      ,args_schema=Weather,
      description="check weathre")
def weather_tool(location: str) -> str:
    return ("The current weather in {location} is sunny with 25°C.")

@tool(name_or_callable = "check_availability"
      ,args_schema=AvailabilityInput,
      description="check_availability")
def check_availability(service: str,
    preferred_datetime: str):
    """
    Check appointment availability.
    """
    return  f"Slot available. Service: {service}, Time: {preferred_datetime}"

@tool(name_or_callable = "extract_booking_details"
      ,args_schema=ExtractBookingInput,
      description="extract_booking_details")
def extract_booking_details(message: str):
    """Extract booking details from user message."""
    data = {}
    services = ["women's cut","men's cut","balayage","blow out","color","keratin"]
    msg = message.lower()
    for service in services:
        if service in msg:data["service"] = service
    phone_match = re.search(r'(\+?\d[\d\s-]{8,15}\d)',message
    )
    if phone_match:
        data["phone"] = phone_match.group()
    return data

@tool(name_or_callable = "create_booking"
      ,args_schema=CreateBookingInput,
      description="create_booking")
def create_booking(service:str,preferred_datetime: str,client_name: str,client_phone: str):
    booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return{
        "booking_id": booking_id,
        "service": service,
        "datetime": preferred_datetime,
        "name": client_name,
        "phone": client_phone,
        "status": "confirmed"
    }

@tool(
    name_or_callable="knowledge_base_search",
    args_schema=RAGInput,
    description="""
    Search salon knowledge base.
    Use for services, pricing, policies,
    opening hours and salon information.
    """)

def rag_search(query: str) -> str:
    """Search salon knowledge base."""
    
    docs = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )