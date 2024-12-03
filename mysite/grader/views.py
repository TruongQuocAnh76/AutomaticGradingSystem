from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# get precomputed rates from csv file
import pandas as pd
import joblib

prompt_rates = np.array(pd.read_csv(os.path.join(current_path, "./../../prompt_vector.csv")))
# training data
X_train = pd.read_csv(os.path.join(current_path, "./../../data_vecs.csv"))
# zscores parameters
params = pd.read_csv(os.path.join(current_path, "./../../precompute.csv"))
# threshold
threshold = -244200212563196.38
threshold1 = -0.22139926189558926

import gensim.downloader as api
model = api.load("word2vec-google-news-300")

# scaler
# scaler = joblib.load("./ss.pkl")
# Create your views here.
def index(request):
    questions_list = Question.objects.order_by('set')
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)

def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')

            if len(content) > 20:
                num_features = 300
                # model = Word2Vec.load(os.path.join(current_path, "deep_learning_files/word2vecmodel.model"))
                # model = Word2Vec.load(os.path.join(current_path, "./../../w2v_otd.model"))
                clean_test_essays = []
                clean_test_essays.append(essay_to_wordlist( content, remove_stopwords=True ))
                testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )
                # testDataVecs = scaler.transform(testDataVecs)

                zscores = modelA(testDataVecs, prompt_rates[question_id - 1].reshape(1, -1), X_train, params.iloc[0, 0], params.iloc[0, 1], params.iloc[0, 2], params.iloc[0, 3])
                if(zscores[0] > threshold and zscores[1] > threshold1):
                    testDataVecs = np.array(testDataVecs)
                    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

                    lstm_model = get_model()
                    lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/final_lstm.h5"))
                    preds = lstm_model.predict(testDataVecs)[0]
                else:
                    preds = -1

                if math.isnan(preds):
                    preds = 0
                else:
                    preds = np.around(preds)

                if preds < 0:
                    preds = 0
                if preds > question.max_score:
                    preds = question.max_score
            else:
                preds = 0

            K.clear_session()
            essay = Essay.objects.create(
                content=content,
                question=question,
                score=preds
            )
        return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "question": question,
        "form": form,
    }
    return render(request, 'grader/question.html', context)