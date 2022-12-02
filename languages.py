import streamlit as st
import i18n
from streamlit.components.v1 import html
from os import abspath, dirname


def change_language():
    lng = st.session_state['lng']

    if lng == 'en':
        i18n.set('locale', 'en')
    else:
        i18n.set('locale', 'es')


def load_text(file_path):
    """A convenience function for reading in the files used for the site's text"""
    with open(file_path) as in_file:
        return in_file.read()


def language_init():
    CURRENT_DIRECTORY = dirname(abspath(__file__))
    
    i18n.load_path.append(CURRENT_DIRECTORY)

    if 'lng' not in st.session_state:
        st.session_state['lng'] = 'en'

    if 'label_lng' not in st.session_state:
        st.session_state['label_lng'] = 'Language'


    globe_icon = load_text('./awesome_globe_icon.html')

    _, _, _, _, col3, col4 = st.columns([3,3,3,3,1,2])
    with col3:
        html(globe_icon, width=25, height=60)
    with col4:
        st.selectbox('', ['en', 'es'], on_change=change_language(), key='lng')


def translate_list(l):
    a_translated = list(map(lambda x: i18n.t(f"translate.{x}"), l))

    return a_translated
