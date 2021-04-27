import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time

st.title('Peer to peer computing')
st.write("Using peer to peer technology to compute large amount of data")

df = pd.DataFrame({
  'pool selection': ['Pool 1','Pool 2','Pool 3','Pool 4'],
})

option = st.sidebar.selectbox(
    'Which pool do you want ?',
     df['pool selection'])

'You selected : ', option

latest_iteration = st.empty()
bar = st.progress(0)

array = [0.0]

if st.button('Compute'):
    chart = st.area_chart(array)
    for i in range(100):
        
        # Random data to display
        latest_iteration.text(f'Computing {i+1}%')
        bar.progress(i + 1)

        df = pd.DataFrame({
            abs((np.random.randn()+10)/10),
        })
        chart.add_rows(df)

        time.sleep(0.1)
    
    st.write("Finished !")
