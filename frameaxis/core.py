from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance
import pkg_resources
import numpy as np
from scipy import stats
from collections import defaultdict
import re
import string

class CoreUtil:
    @staticmethod
    def load_embedding(emb_file, is_binary=True):
        return KeyedVectors.load_word2vec_format(emb_file, binary=is_binary)

    @staticmethod
    def get_score(w_vec, axis_vec):
        return 1.0 - distance.cosine(w_vec, axis_vec)        
    
    @staticmethod
    def map_axis_to_vec(emb, axis):
        return (emb[axis[1]] - emb[axis[0]])

    @staticmethod
    def load_axes_file(filename):
        axes = []
        with open(filename, 'r', encoding='utf-8') as fi:
            for line in fi:
                w1, w2 = [term.strip() for term in line.split("\t")]
                axes.append((w1,w2))                
        return axes 

    @staticmethod
    def load_axes_file_csv(filename, delimiter=',', has_header=False):
        import csv
        axes = []
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            
            # Saltar el encabezado si existe
            if has_header:
                next(csv_reader)
                
            for row in csv_reader:
                if len(row) >= 2:  # Asegurar que hay al menos dos columnas
                    w1, w2 = row[0].strip(), row[1].strip()
                    axes.append((w1, w2))
        
        return axes

    @staticmethod
    def load_conceptnet_antonyms_axes(version = "1.0"):
        if version == "1.0":
            # returns an array of tuples (word1, word2)
            return CoreUtil.load_axes_file(pkg_resources.resource_filename(
                'semaxis', "axes/{}".format('732_semaxis_axes.tsv')))            
        else:
            raise Exception("Unknown version")

    @staticmethod
    def load_plutchik_wheel_of_emotions_axes():
        return CoreUtil.load_axes_file(pkg_resources.resource_filename(
                'semaxis', "axes/{}".format('plutchik_wheel_of_emotions.tsv')))

    @staticmethod
    def load_wordnet_antonyms_axes():
        return CoreUtil.load_axes_file(pkg_resources.resource_filename(
                'semaxis', "axes/{}".format('wordnet_antonyms.tsv')))
    
    @staticmethod
    def load_spanish_antonyms_axes():
        return CoreUtil.load_axes_file(pkg_resources.resource_filename(
                'semaxis', "axes/{}".format('wordnet_spanish_antonyms.tsv')))

    @staticmethod
    def bootstrap_sampling(document, size):
        return np.random.choice(document, size, replace=True)

    @staticmethod
    def _average_dict(data):
        return sum([key*data[key] for key in data])/sum(data.values())
        
    @staticmethod
    def _moment_dict(data, k, data_mean=None):
        if not data_mean:
            data_mean = CoreUtil._average_dict(data)
        return sum(data[key]*(key-data_mean)**k for key in data)/sum(data.values())

    @staticmethod
    def compute_average(axis2score):
        results = []
        averages = {}
        for axis in sorted(axis2score):
            average = CoreUtil._average_dict(axis2score[axis])
            results.append([axis[0], axis[1], average])
            averages[axis] = average

        return results, averages

    @staticmethod
    def compute_kurtosis(axis2score, averages=None):
        results = []
        for axis in sorted(axis2score):
            data = axis2score[axis]
            if averages:
                results.append([axis[0], axis[1], 
                                CoreUtil._moment_dict(data,4,averages[axis])/(CoreUtil._moment_dict(data,2,averages[axis])**2)-3]) 
            else:
                results.append([axis[0], axis[1], 
                                CoreUtil._moment_dict(data,4)/(CoreUtil._moment_dict(data,2)**2)-3]) 
        return results   

    @staticmethod
    def compute_statistical_significance(result, samples):
        samples_dict = defaultdict(list)
        for sample in samples:
            with open(sample) as fi:
                for line in fi:
                    pole0, pole1, value = [term.strip() for term in line.split(",")]
                    try:
                        samples_dict[(pole0, pole1)].append(float(value))
                    except:
                        continue

        axis_rank_table = []
        with open(result) as fi:
            for line in fi:
                pole0, pole1, value = [term.strip() for term in line.split(",")]
                sample_values = samples_dict[(pole0, pole1)]
                try:
                    rank = sum([x < float(value) for x in sample_values])
                except:
                    continue
                percent_rank = rank/float(len(samples))
                axis_rank_table.append([pole0, pole1, percent_rank, value, np.mean(sample_values)])

        return axis_rank_table

    @staticmethod
    def preprocess(document):
        def remove_punctuation(text):
            punctuation = string.punctuation + "¿¡"

            # Crear una tabla de traducción que reemplace los signos de puntuación por un espacio
            translator = str.maketrans({char: ' ' for char in punctuation})

            return text.translate(translator)

        def remove_numbers(text):
            numbers_re = re.compile(r'\d+')
            return numbers_re.sub(' ', text)

        def remove_urls(text):
            urls_re = re.compile(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')
            return urls_re.sub(' ', text)

        def remove_spaces_newlines_tabs(text):
            # Cleans text by removing whitespace, newlines and tabs
            return " ".join(text.strip().split())


        # remover urls, puntuacion, numeros, mayusculas y espacios/tabs/saltos de linea
        document = remove_urls(document)
        document = remove_punctuation(document)
        document = remove_numbers(document)
        document = document.lower()
        document = remove_spaces_newlines_tabs(document)

        return document