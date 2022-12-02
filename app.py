import streamlit as st
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import i18n

from languages import language_init


def show_graph(x, y, width, height, title, m_pred=None, b_pred=None):
    
    fig, ax = plt.subplots(figsize=(width, height))

    plt.axhline(y=0, linestyle='--', c='lightgrey', zorder=-1)
    plt.axvline(x=0, linestyle='--', c='lightgrey', zorder=-1)

    ax.scatter(x, y)

    if (m_pred):
        y_pred = (m_pred * x + b_pred).detach().numpy()
        ax.plot(x, y_pred, c='r')

    plt.title(title)
    plt.xlabel(i18n.t('translate.Fluid'))
    plt.ylabel(i18n.t('translate.Temperature'))
    plt.xlim([-1, 11])
    plt.ylim([-2, 11])

    st.pyplot(fig)


def show_metrics(m, b, cost):
    col1, col2, col3 = st.columns(3)
    col1.metric(i18n.t('translate.Slope'), round(m, 3))
    col2.metric(i18n.t('translate.Intercept'), round(b, 3))
    col3.metric(i18n.t('translate.Cost'), round(cost, 3))


@st.cache
def sgd(x, y, m_init, b_init, epochs):
    m_pred = torch.tensor([m_init]).requires_grad_()
    b_pred = torch.tensor([b_init]).requires_grad_()
    optimizer = torch.optim.SGD([m_pred, b_pred], lr=0.01)

    m_pred_v = []
    b_pred_v = []
    Cost_v = []
    epochs = 1000
    for epoch in range(epochs):

        optimizer.zero_grad()
        
        y_pred = m_pred * x + b_pred
        Cost = mse_loss(y_pred, y)
        Cost.backward()

        optimizer.step()

        m_pred_v.append(m_pred.detach().numpy().item())
        b_pred_v.append(b_pred.detach().numpy().item())
        Cost_v.append(Cost.item())

    return m_pred_v, b_pred_v, Cost_v


 
language_init()

#st.write(i18n.get('locale'))
#st.write(i18n.load_path)
st.write(open('/app/translate.en.yml').read())

st.markdown(i18n.t('translate.title'), unsafe_allow_html=True)
st.write('\n')

N = 10
WIDTH_FIG = 7
HEIGHT_FIG = 4
EPOCHS = 1000

# Building data to predict
x = torch.tensor(np.arange(1., N + 1))

m = -0.6
b = 5.
std = 0.4
random_seed = 42
torch.manual_seed(random_seed)
y = m*x + b + torch.normal(mean=torch.zeros(N), std=std)

df = pd.DataFrame(y.numpy().reshape(1, N), 
                    columns=map(lambda x: str(x) + ' mL', np.arange(1, N + 1)),
                    index=["°C"])
st.markdown(i18n.t('translate.intro_1'))
st.markdown(i18n.t('translate.intro_2'))
st.table(df)

# Show data to predict
show_graph(x, y, WIDTH_FIG, HEIGHT_FIG, i18n.t('translate.title_graph_1'))

st.markdown(i18n.t('translate.comment_1_graph_1'))
st.markdown(i18n.t('translate.comment_2_graph_1'))
st.markdown(i18n.t('translate.comment_3_graph_1'))
st.markdown(i18n.t('translate.comment_4_graph_1'))
st.markdown("\n")

# 'm' and 'b' random initial values for applying SGD method
st.markdown(i18n.t('translate.comment_initial_values'))
m_init = 0.9
b_init = 0.1
y_init = m_init * x + b_init
Cost = mse_loss(y_init, y)

show_metrics(m_init, b_init, Cost.item())
show_graph(x, y, WIDTH_FIG, HEIGHT_FIG, i18n.t('translate.title_graph_2'), m_init, b_init)

# SGD method
m_pred_v, b_pred_v, Cost_v = sgd(x, y, m_init, b_init, EPOCHS)

st.markdown(i18n.t('translate.comment_SGD'))
epoch = st.slider(i18n.t('Epochs'), 1, EPOCHS)

m_epoch = m_pred_v[epoch-1]
b_epoch = b_pred_v[epoch-1]
Cost_epoch = Cost_v[epoch-1]

show_metrics(m_epoch, b_epoch, Cost_epoch)
show_graph(x, y, WIDTH_FIG, HEIGHT_FIG, i18n.t('translate.title_graph_3'), m_epoch, b_epoch)
 
