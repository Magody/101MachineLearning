import requests
import io
import pandas as pd
import json

class APIOpenAI:
    def __init__(self, api_key, verbose=1, auto_parse_response=False):
        self.api_key = api_key
        self.verbose = verbose
        self.auto_parse_response = auto_parse_response

    def log(self, msg, verbose_minimum=0):
        """
        Centralized place to print messages according to verbose

        Parameters:
          msg (str): the string message
          verbose_minimum(int,Optional): The minimum verbose needed to print the message
        Returns:
          None
        """
        if self.verbose >= verbose_minimum:
            print(msg)

    def _add_if_not_none(self, dictionary, name, value):
      if value is not None:
        dictionary[name] = value
      return dictionary

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_url(self, endpoint):
        return f"https://api.openai.com/v1/{endpoint}"

    def request(self, url, method="GET", data={}, files=None):
        """
        HTTP request made to OpenAI api

        Parameters:
          params (dict): url params to send to the api
          url(str): Endpoint where we can make the request for a <report-type> and an <organic/inorganic> option
        Returns:
          (dict) Result
        """
        self.log(f"{method} a {url}", 5)
        if method.upper() == 'GET':
          res = requests.get(url, params=data, headers=self._get_headers())
        elif method.upper() == 'POST':
          if files is None:
            res = requests.post(url, data=json.dumps(data), headers=self._get_headers())
          else:
            headers = self._get_headers()
            del headers["Content-Type"]
            res = requests.post(url, files=files, headers=headers)
        elif method.upper() == 'DELETE':
          res = requests.delete(url, headers=self._get_headers())
        else:
          raise Exception(f"Method not supported: {method}")
        
        self.response = res

        if res.status_code != 200:
            if res.status_code == 404:
                self.log(
                    "There is a problem with the request URL. Make sure that it is correct"
                )
            else:
                self.log("There was a problem retrieving data: " + res.text)

        
        if self.auto_parse_response:
            try:
              return res.json()
            except Exception as error:
              return res.text
        return res

    def get_models(self):
        """
        GET /models
        """
        return self.request(url=self._get_url("models"))

    def get_specific_model(self, model):
        """
        GET /models/{model}
        """
        return self.request(url=self._get_url(f"models/{model}"))

    def create_completion(
        self,
        model,
        prompt,
        suffix=None,
        max_tokens=16,
        temperature=1,
        top_p=1,
        n=1,
        stream=False,
        logprobs=None,
        echo=False,
        stop=None,
        presence_penalty=0,
        frequency_penalty=0,
        best_of=1
    ):
        """
        POST /completions
        openai_api.create_completion("text-davinci-001", "Hello, what is the capital of Ecuador?", logprobs=3)
        """
        body = {
          "model": model,
          "prompt": prompt,
          "suffix": suffix,
          "max_tokens": max_tokens,
          "temperature": temperature,
          "top_p": top_p,
          "n": n,
          "stream": stream,
          "logprobs": logprobs,
          "echo": echo,
          "stop": stop,
          "presence_penalty": presence_penalty,
          "frequency_penalty": frequency_penalty,
          "best_of": best_of
        }
        return self.request(
            url=self._get_url(f"completions"), method="POST", data=body
        )

    def create_chat_completion(
        self,
        model,
        messages,
        temperature=1,
        top_p=1,
        n=1,
        stream=False,
        stop=None,
        max_tokens=None,
        presence_penalty=0,
        frequency_penalty=0
    ):
        """
        POST /chat/completions
        openai_api.create_chat_completion(
          "gpt-3.5-turbo", [
            {"role": "system", "content": "Eres un asistente que vende firmas electrónicas, el precio de una firma va desde 30 dólares a 50 dólares. Lo mejor es que le ofrescas primero un precio alto con la posibilidad de un descuento"},
            {"role": "user", "content": "hola, cuanto cuesta una firma electrónica?!"}
          ]
        )
        """
        body = {
          "model": model,
          "messages": messages,
          "temperature": temperature,
          "top_p": top_p,
          "n": n,
          "stream": stream,
          "stop": stop,
          "max_tokens": max_tokens,
          "presence_penalty": presence_penalty,
          "frequency_penalty": frequency_penalty
        }
        return self.request(
            url=self._get_url(f"chat/completions"), method="POST", data=body
        )

    def create_edit(
        self,
        model,
        instruction,
        input_starting_point='',
        temperature=1,
        top_p=1,
        n=1
    ):
        """
        POST /edits
        openai_api.create_edit("text-davinci-edit-001", "Fix the spelling mistakes", "What day of the wek is it?")
        """
        body = {
          "model": model,
          "instruction": instruction,
          "input": input_starting_point,
          "temperature": temperature,
          "top_p": top_p,
          "n": n,
        }
        return self.request(
            url=self._get_url(f"edits"), method="POST", data=body
        )
    
    def generate_image(
        self,
        prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    ):
        """
        POST /images/generations
        openai_api.generate_image("un dashboard de un ingeniero de datos")
        """
        body = {
          "prompt": prompt,
          "n": n,
          "size": size,
          "response_format": response_format,
        }
        return self.request(
            url=self._get_url(f"images/generations"), method="POST", data=body
        )

    def generate_image_edit(self):
        raise Exception("Not implemented")

    def generate_image_variation(self):
        raise Exception("Not implemented")

    def create_embeddings(self, model, input_text_or_array):
        """
        POST /embeddings
        openai_api.create_embeddings("text-embedding-ada-002", "The food was delicious and the waiter...")
        """
        body = {
          "model": model,
          "input": input_text_or_array,
        }
        return self.request(
            url=self._get_url(f"embeddings"), method="POST", data=body
        )

    def create_audio_transcription(self):
        raise Exception("Not implemented")

    def create_audio_translation(self):
        raise Exception("Not implemented")

    def get_files(self):
        """
        GET /files
        """
        return self.request(url=self._get_url("files"))

    def upload_file(self, file_path, purpose='fine-tune'):
        """
        POST /files
        openai_api.upload_file("/dbfs/tmp/openai/intentions_train.jsonl")
        """
        files = {
          "file": (file_path.split("/")[-1], open(file_path, 'rb')),
          'purpose': (None, purpose),
        }
        return self.request(url=self._get_url("files"), method='POST', files=files)

    def delete_file(self, file_id):
        """
        DELETE /files/{file_id}
        """
        return self.request(url=self._get_url(f"files/{file_id}"), method='DELETE')
    
    def get_specific_file(self, file_id):
        """
        GET /files/{file_id}
        """
        return self.request(url=self._get_url(f"files/{file_id}"))

    def get_file_content(self, file_id):
        """
        GET /files/{file_id}/content
        """
        return self.request(url=self._get_url(f"files/{file_id}/content"))

    def create_fine_tune(
        self,
        training_file,
        validation_file=None,
        model='curie',
        n_epochs=4,
        batch_size=None,
        learning_rate_multiplier=None,
        prompt_loss_weight=0.01,
        compute_classification_metrics=False,
        classification_n_classes=None,
        classification_positive_class=None,
        classification_betas=None,
        suffix=None
    ):
        """
        POST /fine-tunes
        openai_api.create_fine_tune("file-Yfdyo6UaZPTtnatsB7TX47Xh", model='ada', batch_size=16)
        """
        body = {
          "training_file": training_file,
          "model": model,
          "n_epochs": n_epochs,
          "batch_size": batch_size,
          "prompt_loss_weight": prompt_loss_weight,
          "compute_classification_metrics": compute_classification_metrics
        }
        body = self._add_if_not_none(body, "validation_file", validation_file)
        body = self._add_if_not_none(body, "batch_size", batch_size)
        body = self._add_if_not_none(body, "learning_rate_multiplier", learning_rate_multiplier)
        body = self._add_if_not_none(body, "classification_n_classes", classification_n_classes)
        body = self._add_if_not_none(body, "classification_positive_class", classification_positive_class)
        body = self._add_if_not_none(body, "classification_betas", classification_betas)
        body = self._add_if_not_none(body, "suffix", suffix)
        return self.request(
            url=self._get_url(f"fine-tunes"), method="POST", data=body
        )
    
    def get_fine_tunes(self):
        """
        GET /fine-tunes
        """
        return self.request(url=self._get_url("fine-tunes"))
    
    def get_specific_fine_tunes(self, fine_tune_id):
        """
        GET /fine-tunes/{fine_tune_id}
        """
        return self.request(url=self._get_url(f"fine-tunes/{fine_tune_id}"))
    
    def cancel_fine_tunes(self, fine_tune_id):
        """
        POST /fine-tunes/{fine_tune_id}/cancel
        """
        return self.request(url=self._get_url(f"fine-tunes/{fine_tune_id}/cancel"), method='POST')
    
    def get_fine_tunes_events(self, fine_tune_id, stream=False):
        """
        GET /fine-tunes/{fine_tune_id}/events
        """
        params = {
          "stream": stream
        }
        return self.request(url=self._get_url(f"fine-tunes/{fine_tune_id}/events"), data=params)
    
    def delete_fine_tunes(self, model):
        """
        DELETE /models/{model}
        """
        return self.request(url=self._get_url(f"models/{model}"), method='DELETE')
    
    def create_moderation(self, input_text, model='text-moderation-latest'):
        """
        POST /moderations
        """
        body = {
          "input": input_text,
          "model": model
        }
        return self.request(url=self._get_url("moderations"), method='POST', data=body)    