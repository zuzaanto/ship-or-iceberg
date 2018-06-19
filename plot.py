import plotly.offline as py
import plotly.graph_objs as go

#py.init_notebook_mode(connected=True)
def plotmy3d(c, name):

    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65, 
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
def iceorship(no):
    if target_train[no]==0:
        return "statek"
    else:
        return "góra lodowa"
train = pandas.read_json("./data/processed/train.json")

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
target_train=train['is_iceberg']
plotmy3d(X_band_1[102,:,:], iceorship(102))