import pandas as pd
import numpy as np
import re



def is_in_stop_list(word):
  # функция возвращает True, если слово на входе часто в трейне пишется с большой буквы 
  stop_list = ['заказчик', 'получател', 'договор', 'поставщик', 
      'федер', 'закон', 'росси', 'гражданск', 
      'министерств', 'директор', 'сторон', 'оборудован', 
      'акт', 'мвд', 'нмцк', 'удмур', 'глав', 'москв', 'контракт', 
      'исполнител', 'ндс', 'правительств', 'рф', 'документ', 
      'продукци', 'проект']

  for exception in stop_list:
    word = word.lower()
    if word.startswith(exception):
      return True
  return False


def is_common_target_end(sentence):
  # возвращает индексы частых окончаний фрагментов
  # в стоп листе сохраним падежи, чтобы не генерить много фрагментов
  stop_list = [
      'контракта', 'нцмд', 'копейки', 'договора', 'рублей', '(цены лота)'
  ]

  index_list = []

  word_list = sentence.split()
  for word in word_list:
    if word.lower() in stop_list:
      index_list.extend([m.start() + len(word) for m in re.finditer(word,  sentence)])


  return list(set(index_list))


def comma_separate(list_of_sentences, min_sentence_lenght=15):
  # делит полученный список предложений на фрагменты по запятым, тире и частым окончаниям, возвращает полный список предложений с фрагментами 
  result_fragments_list = []

  def separate(sentence):
    list_of_commas_index = []
    list_of_comma_separated_fragments = []
    for i in range(len(sentence)):
      try:
        if sentence[i] == ',' or sentence[i] == '-':
          if not sentence[i-1].isnumeric() and not sentence[i-1].isupper() and not sentence[i+1].isnumeric() and not sentence[i+1].isupper():
            list_of_commas_index.append(i)
      except:
        pass

    list_of_commas_index.extend(is_common_target_end(sentence))
    start = 0
    for end in list_of_commas_index:
      fragment = sentence[start:end]
      list_of_comma_separated_fragments.append(fragment)
    
    return list(set(list_of_comma_separated_fragments))

  
  for sentence in list_of_sentences:
    if len(sentence) > min_sentence_lenght:
      result_fragments_list.extend(separate(sentence))

  return result_fragments_list


def check_the_last_word(list_of_words):
  # проверяет не является ли последнее слово в предложении знаком > < или небольшим числом
  try: 
    if int(list_of_words[-1]) < 100:
      return list_of_words[:-1]
  except:
    if '<' in list_of_words[-1] and '>' in list_of_words[-1]:
      return list_of_words[:-1]
  return list_of_words


def splitter(text, min_sentence_lenght=6):
  # функция разделения текстов на предложения и фрагменты, возвращает их список
  list_of_sentences = []
  list_of_words = text.split()
  sentence = []
  previous_word = ''
  for word in list_of_words:
    try:
      if not len(word) == 1:
        # критерии начала следующего предложения
        if word[-1] == '.' and word[-2].isnumeric():
          sentence = check_the_last_word(sentence)
          list_of_sentences.append(' '.join(sentence))
          sentence = []
        if word[0].isupper() and not word.isupper():
          if not is_in_stop_list(word) and sentence:
            sentence = check_the_last_word(sentence)
            list_of_sentences.append(' '.join(sentence))
            sentence = []
        if is_in_stop_list(word) and previous_word[-1] == '.':
          sentence = check_the_last_word(sentence)
          list_of_sentences.append(' '.join(sentence))
          sentence = []

      else:
        if word.isupper():
          sentence = check_the_last_word(sentence)
          list_of_sentences.append(' '.join(sentence))
          sentence = []

    except:
      pass
    sentence.append(word)

    if word is list_of_words[-1]:
      sentence = check_the_last_word(sentence)
      list_of_sentences.append(' '.join(sentence))
    previous_word = word

  # добавим фрагменты к списку предложений если extended=True
  # if extended:
  list_of_sentences.extend(comma_separate(list_of_sentences))

  for sentence in list_of_sentences:
    if len(sentence) < min_sentence_lenght or not sentence:
      del list_of_sentences[list_of_sentences.index(sentence)]
  return list_of_sentences
      


def division_check(df:pd.DataFrame):
  # подсчитывает качество разделения текстов на фрагменты
  counter = 0
  try:
    df.drop(columns=['not_found'], inplace=True)
  except:
    pass

  for i in range(len(df)):
    try:
      if df.loc[i, 'target'] in splitter(df.loc[i, 'text']):
        counter += 1
      else:
        df.loc[i, 'not_found'] = 0
    except IndexError:
      print(df.loc[i, 'text'])
      print(i)
  return counter / 1492, df[df['not_found']==0].reset_index(drop=True)  # 1492 - ненулевые значения в трейне



def split_df_texts_by_sentences(data_frame:pd.DataFrame):
  # разделяет тексты в датафрейме на предложения, на выходе датафрейм с разделёнными текстами
  for i in range(len(data_frame)-1):
    splitted = splitter(data_frame.loc[i, 'text'])
    data_frame.at[i, 'text'] = splitted
  return data_frame



def text_accuracy_score(prediction, real_data, precent=False):
    # счиатет accuracy предсказания
    assert len(prediction) == len(real_data), 'Objects must have the same length'
    counter = 0

    for pred, real in zip(prediction, real_data):
        if pred == real:
          counter += 1

    score = counter / len(real_data)
    if precent:
      return f'{score * 100}%'

    return score



def choose_minimum_rate_sentence(result:dict, min_rate=None):
  # принимает на вход словарь вида: {'sentence1': 0.011, 'sentence2': 0.6} 
  # возвращает sentence1, 0.011 (минимальное значение)
  min_val = 1e4
  for key, value in result.items():
    if value < min_val:
      min_val = value
      sentence = key


  if min_rate:
    if min_val < min_rate:
      return sentence, min_val
    else:
      return None, None

  return sentence, min_val



def choose_max_rate_sentence(result:dict, max_rate=None):
  # принимает на вход словарь вида: {'sentence1': 0.011, 'sentence2': 0.6}
  # возвращает sentence2, 0.6 (максимальное значение)
  max_val = -10
  for key, value in result.items():
    if value > max_val:
      max_val = value
      sentence = key
      
  if max_rate:
    if max_val > max_rate:
      return sentence, max_val
    else:
      return None, None  

  return sentence, max_val


def lemmatization(text, stopwords, morph):
  # предит предложения в список токенов
  if text and len(text) > 1:
    pattern = "[A-Za-z0-9!#$%&'()*+№,./:;<=>?@[\]^_`{|}~—\"\-]+"
    text = re.sub(pattern, ' ', text)
    tokens = []
    for token in text.split():
        if token and token not in stopwords:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    return tokens
  else: 
    return None



def get_word_vectors_from_dataframe(dataframe, navec):
  # переводит лемматизированный датафрейм в датафейм из векторов предложенний, возвращает датафрейм текстов из векторов предложений
  df = []
  for i in range(len(dataframe)):
    sentence = []
    for j in range(len(dataframe.loc[i, 'text'])):
      try:
        word_embending = navec[dataframe.loc[i, 'text'][j]].mean()
        sentence.append(word_embending)
      except:
        sentence.append(0)
    df.append(sentence)
  df = pd.DataFrame(df, dtype='float64')
  df['target'] = dataframe['target']
  df.fillna(0, inplace=True)
  return df


def split_df_texts_by_sentences(data_frame:pd.DataFrame):
  # делит тексты датафрейма функцией splitter
  for i in range(len(data_frame)):
    splitted = splitter(data_frame.loc[i, 'text'])
    data_frame.at[i, 'text'] = splitted
  return data_frame



def lemmatize_splitted_df_sentences(dataframe:pd.DataFrame, stopwords, morph):
  """
  приводит разделённые тексты датафрема к формату: 
  [['не', 'установить'], ['обеспечение', 'заявка'], ['обеспечение', 'исполнение', 'контракт']]
  возвращает датафрейм
  """
  for i in range(len(dataframe)-1):
    sentence = []
    for k in range(len(dataframe.loc[i, 'text'])):
      lemma = lemmatization(dataframe.loc[i, 'text'][k], stopwords=stopwords, morph=morph)
      sentence.append(lemma)
    dataframe.at[i, 'text'] = sentence
  return dataframe



def get_sentence_vectors_from_lemmatized_text(text:list, vector_len, navec):
  # переводит лемматизированный текст в список векторов заданной длины для применения модели
  vectorized_text = []
  for i in range(len(text)):
    sentence = [0 for _ in range(vector_len)]
    if text[i] and len(text[i]) > vector_len:
      for k in range(vector_len):
        try: 
          word_emdending = navec[text[i][k]].mean()
          sentence[k] = word_emdending
        except:
          sentence[k] = 0
      vectorized_text.append(sentence)

    elif text[i] is None:
      vectorized_text.append(sentence)
      continue

    else: 
      for k in range(len(text[i])):
        try: 
          word_emdending = navec[text[i][k]].mean()
          sentence[k] = word_emdending
        except:
          sentence[k] = 0
      vectorized_text.append(sentence)

  return vectorized_text



def get_extracted_part_from_dataset(dataframe, predictions):
  # формирует extracted_part для предсказания модели
  result = []

  # пробегаем по предсказанным предложениям
  for sentence, text in zip(predictions, dataframe.loc[:, 'text'].values):
    query = {
      "text": [""],
      "answer_start": [0],
      "answer_end": [0]
        }
    if sentence is not None:
      try:
        answer_position = list(re.finditer(re.escape(sentence), text))[0]
        answer_start = [answer_position.start()]
        answer_end = [answer_position.end()]
        text = [sentence]

        query['text'] = text
        query['answer_start'] = answer_start
        query['answer_end'] = answer_end
      except Exception as e:
        print(f'text: {text}, \nprediction: {sentence}\nexception: {e}\nanswer_position:{list(re.finditer(sentence, text))}')
        print()
        print()
        result.append(query)
        continue

    result.append(query)
  return result








