# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.767390Z","iopub.execute_input":"2025-03-01T05:41:53.767865Z","iopub.status.idle":"2025-03-01T05:41:53.912328Z","shell.execute_reply.started":"2025-03-01T05:41:53.767823Z","shell.execute_reply":"2025-03-01T05:41:53.910965Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# from scipy import stats

df = pd.read_csv("AB_NYC_2019.csv")
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.913403Z","iopub.execute_input":"2025-03-01T05:41:53.913662Z","iopub.status.idle":"2025-03-01T05:41:53.921300Z","shell.execute_reply.started":"2025-03-01T05:41:53.913633Z","shell.execute_reply":"2025-03-01T05:41:53.919714Z"},"jupyter":{"outputs_hidden":false}}
print(df.shape)
print(df.dtypes)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.923889Z","iopub.execute_input":"2025-03-01T05:41:53.924210Z","iopub.status.idle":"2025-03-01T05:41:53.997642Z","shell.execute_reply.started":"2025-03-01T05:41:53.924188Z","shell.execute_reply":"2025-03-01T05:41:53.995462Z"},"jupyter":{"outputs_hidden":false}}
df[df.duplicated()].sum()  # Dataset Duplicate Value Count

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.999348Z","iopub.execute_input":"2025-03-01T05:41:54.000901Z","iopub.status.idle":"2025-03-01T05:41:54.029426Z","shell.execute_reply.started":"2025-03-01T05:41:54.000801Z","shell.execute_reply":"2025-03-01T05:41:54.028091Z"},"jupyter":{"outputs_hidden":false}}
df.isna().sum() #null entries

