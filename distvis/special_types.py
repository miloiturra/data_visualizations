import pandas as pd
from typing import Callable, Union

AggFunction = Callable[[pd.DataFrame], Union[float, int]]
