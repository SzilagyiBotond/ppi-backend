from pydantic import BaseModel
from typing import Dict, List

class ModelResponse(BaseModel):
    analysis_id: str
    plots: Dict[str, bytes]  # plot_name -> plot_image