from torchsig.datasets.narrowband import StaticNarrowband
import click
import plotly.graph_objects as go
import numpy as np

dataset = StaticNarrowband(root='data', impaired=False)

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(dataset[0][0])))
fig.add_trace(go.Scatter(y=np.imag(dataset[0][0])))
# fig = px.line(dataset[0][0], title=dataset[0][1][0]['class_name'])
fig.update_layout(title=dataset[0][1][0]['class_name'])
fig.show()


