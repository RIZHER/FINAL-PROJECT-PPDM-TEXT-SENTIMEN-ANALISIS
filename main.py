from lib import *
from model import *
from preprocessing import *


def main():

    st.title("Final Projek PPDM")
    st.title(
        ":green[Review Ulasan(Sentiment Analysis Expression)]:blue[dengan Python]")
    st.header('Menerapkan Metode Multinomial Naive Bayes And Chi - Square')
    teks = st.text_input('Input text')
    hasil = preprocess_text(teks)
    kalimat_normalisasi = normalized_term(hasil)

    def convert_to_sentence(word_list):
        sentence = ' '.join(word_list)
        return sentence
    kalimat = convert_to_sentence(kalimat_normalisasi)

    if st.button('Prediksi Kalimat'):
        prediction = predict_text(kalimat)


if __name__ == '__main__':
    main()
