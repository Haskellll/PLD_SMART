import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import tkinter as tk  
from tkinter import filedialog  

st.title('Peer to peer computing')
st.write("Using peer to peer technology to compute large amount of data")


option = 'Pool 0'

st.header('\nPool selection :')

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

st.header('\nJob file upload :')

# TODO : add more file extensions
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'py'])

if st.button('Upload'):
    if uploaded_file is not None:
        st.write('Uploaded !')