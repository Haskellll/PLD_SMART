import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('Peer to peer computing')
st.write("Using peer to peer technology to compute large amount of data")


option = 'Pool 0'


table = pd.DataFrame({
    'Pools': ['Pool 1','Pool 2','Pool 3','Pool 4'],
    'Active nodes': [10, 20, 30, 40],
    'CPU usage': ['10%', '20%', '30%', '40%']
})
            
st.write(table)

option = st.selectbox(
    'Which pool do you want ?',
    table['Pools'])
'You selected : ', option