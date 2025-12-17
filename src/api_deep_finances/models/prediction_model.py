from sqlalchemy import Column, String, Date, Time, Numeric
from sqlalchemy.sql import func
from configs.database import Base

class Prediction(Base):
    __tablename__ = "previsions"
    
    id = Column(String, primary_key=True, index=True, autoincrement=False)
    ticker = Column(String, nullable=False)
    prevision_price = Column(Numeric(11, 2), nullable=False)
    collection_date = Column(Date, server_default=func.current_date())
    collection_time = Column(Time, server_default=func.current_time())
    