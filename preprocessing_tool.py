# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.767390Z","iopub.execute_input":"2025-03-01T05:41:53.767865Z","iopub.status.idle":"2025-03-01T05:41:53.912328Z","shell.execute_reply.started":"2025-03-01T05:41:53.767823Z","shell.execute_reply":"2025-03-01T05:41:53.910965Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

df = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.913403Z","iopub.execute_input":"2025-03-01T05:41:53.913662Z","iopub.status.idle":"2025-03-01T05:41:53.921300Z","shell.execute_reply.started":"2025-03-01T05:41:53.913633Z","shell.execute_reply":"2025-03-01T05:41:53.919714Z"},"jupyter":{"outputs_hidden":false}}
print(df.shape)
print(df.dtypes)

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.923889Z","iopub.execute_input":"2025-03-01T05:41:53.924210Z","iopub.status.idle":"2025-03-01T05:41:53.997642Z","shell.execute_reply.started":"2025-03-01T05:41:53.924188Z","shell.execute_reply":"2025-03-01T05:41:53.995462Z"},"jupyter":{"outputs_hidden":false}}
df[df.duplicated()].sum()  # Dataset Duplicate Value Count

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:53.999348Z","iopub.execute_input":"2025-03-01T05:41:54.000901Z","iopub.status.idle":"2025-03-01T05:41:54.029426Z","shell.execute_reply.started":"2025-03-01T05:41:54.000801Z","shell.execute_reply":"2025-03-01T05:41:54.028091Z"},"jupyter":{"outputs_hidden":false}}
df.isna().sum() #null entries

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:54.030694Z","iopub.execute_input":"2025-03-01T05:41:54.031110Z","iopub.status.idle":"2025-03-01T05:41:54.090006Z","shell.execute_reply.started":"2025-03-01T05:41:54.031074Z","shell.execute_reply":"2025-03-01T05:41:54.088308Z"},"jupyter":{"outputs_hidden":false}}
df.nunique()

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:54.091022Z","iopub.execute_input":"2025-03-01T05:41:54.091381Z","iopub.status.idle":"2025-03-01T05:41:54.137084Z","shell.execute_reply.started":"2025-03-01T05:41:54.091349Z","shell.execute_reply":"2025-03-01T05:41:54.135163Z"},"jupyter":{"outputs_hidden":false}}
df.describe()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Identifying Missing Values

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:54.138010Z","iopub.execute_input":"2025-03-01T05:41:54.138326Z","iopub.status.idle":"2025-03-01T05:41:54.187190Z","shell.execute_reply.started":"2025-03-01T05:41:54.138298Z","shell.execute_reply":"2025-03-01T05:41:54.185382Z"},"jupyter":{"outputs_hidden":false}}
print(df.isnull().sum())  # Count of missing values per column
print("\n\n")
print(df.isnull().any())  # Boolean: Whether each column has missing values
print("\n\n")
print(df.isnull().sum().sum())  # Total count of missing values

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:54.189661Z","iopub.execute_input":"2025-03-01T05:41:54.190159Z","iopub.status.idle":"2025-03-01T05:41:55.942899Z","shell.execute_reply.started":"2025-03-01T05:41:54.190055Z","shell.execute_reply":"2025-03-01T05:41:55.941056Z"},"jupyter":{"outputs_hidden":false}}
import missingno as msno
msno.bar(df)
msno.matrix(df)

# %% [markdown] {"execution":{"iopub.status.busy":"2025-03-01T05:26:16.756288Z","iopub.execute_input":"2025-03-01T05:26:16.756607Z","iopub.status.idle":"2025-03-01T05:26:16.760245Z","shell.execute_reply.started":"2025-03-01T05:26:16.756583Z","shell.execute_reply":"2025-03-01T05:26:16.759375Z"},"jupyter":{"outputs_hidden":false}}
# # Identifying Duplicates

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:55.945024Z","iopub.execute_input":"2025-03-01T05:41:55.945405Z","iopub.status.idle":"2025-03-01T05:41:56.058052Z","shell.execute_reply.started":"2025-03-01T05:41:55.945368Z","shell.execute_reply":"2025-03-01T05:41:56.056681Z"},"jupyter":{"outputs_hidden":false}}
# Detect duplicate rows
duplicates = df.duplicated()
print(duplicates)

# Count duplicates
print(df.duplicated().sum())

# Drop duplicates if needed
df_cleaned = df.drop_duplicates()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **Check distribution of categorical variables**

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.059086Z","iopub.execute_input":"2025-03-01T05:41:56.059366Z","iopub.status.idle":"2025-03-01T05:41:56.069880Z","shell.execute_reply.started":"2025-03-01T05:41:56.059339Z","shell.execute_reply":"2025-03-01T05:41:56.067592Z"},"jupyter":{"outputs_hidden":false}}
print(df['room_type'].value_counts(normalize=True))  # Normalize=True gives proportions

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **Check for feature correlations (to detect potential biases in data relationships):**

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.070982Z","iopub.execute_input":"2025-03-01T05:41:56.071417Z","iopub.status.idle":"2025-03-01T05:41:56.095692Z","shell.execute_reply.started":"2025-03-01T05:41:56.071375Z","shell.execute_reply":"2025-03-01T05:41:56.093400Z"},"jupyter":{"outputs_hidden":false}}
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Identifying Class Imbalance

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.097492Z","iopub.execute_input":"2025-03-01T05:41:56.097941Z","iopub.status.idle":"2025-03-01T05:41:56.275031Z","shell.execute_reply.started":"2025-03-01T05:41:56.097906Z","shell.execute_reply":"2025-03-01T05:41:56.273859Z"},"jupyter":{"outputs_hidden":false}}
# Count the instances in each class
print(df['room_type'].value_counts())

# Plot class distribution
import seaborn as sns
sns.countplot(x=df['room_type'])

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:42:17.693807Z","iopub.execute_input":"2025-03-01T05:42:17.694102Z","iopub.status.idle":"2025-03-01T05:42:17.714515Z","shell.execute_reply.started":"2025-03-01T05:42:17.694078Z","shell.execute_reply":"2025-03-01T05:42:17.712662Z"},"jupyter":{"outputs_hidden":false}}
# For more advanced detection:
from imblearn.over_sampling import SMOTE

y = df['room_type']
X = df.drop(columns=['room_type'])

smote = SMOTE #For more advanced detection:

X_resampled, y_resampled = smote.fit_resample(X, y)

print(y_resampled.value_counts())  # Now balanced

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Identifying Outliers

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.296391Z","iopub.status.idle":"2025-03-01T05:41:56.296950Z","shell.execute_reply":"2025-03-01T05:41:56.296710Z"},"jupyter":{"outputs_hidden":false}}
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3)  # Mark values beyond 3 standard deviations
print(df[outliers.any(axis=1)])  # Rows with outliers

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.298004Z","iopub.status.idle":"2025-03-01T05:41:56.298332Z","shell.execute_reply":"2025-03-01T05:41:56.298213Z"},"jupyter":{"outputs_hidden":false}}
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
print(df[outliers.any(axis=1)])  # Display rows with outliers

# %% [code] {"execution":{"iopub.status.busy":"2025-03-01T05:41:56.298933Z","iopub.status.idle":"2025-03-01T05:41:56.299205Z","shell.execute_reply":"2025-03-01T05:41:56.299088Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt

plt.boxplot(df['A'].dropna())  # Column 'A'
plt.show()