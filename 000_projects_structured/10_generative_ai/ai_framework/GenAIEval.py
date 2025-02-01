from ai_framework.LoggingAndTelemetry import EnumLogs, Logging
from bert_score import score
from mlflow.metrics.genai import (
    # EvaluationExample,
    answer_correctness,
    answer_similarity,
    faithfulness,
    answer_relevance,
    relevance,
)
from mlflow.metrics import (
    # exact_match,
    ari_grade_level,
    flesch_kincaid_grade_level,
    latency,
    rouge2,
    rougeL,
    rougeLsum,
    # toxicity,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)
from enum import Enum
from typing_extensions import List
import pandas as pd
from mlflow import MlflowException
from mlflow.metrics.genai import model_utils
from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    # EnumGenAIModelsIdsOpenAI,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import GenAIPrompt
import os
import mlflow


def _load_model_or_server(model_uri, payload):
    """
    Load the model identified by the given URI and use it to predict on the given payload.

    Args:
        model_uri (str): The URI of the model to load.
        payload (any): The input data to pass to the model.

    Returns:
        The prediction result from the loaded model.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model.predict(payload)


def score_model_on_payload(model_uri, payload, eval_parameters=None):
    """
    Call the model identified by the given URI with the given payload.

    Args:
        model_uri (str): The URI of the model to call.
        payload (any): The input data to pass to the model.
        eval_parameters (dict, optional): Additional parameters to pass to the model. Defaults to None.

    Returns:
        The result of calling the model with the given payload.
    """

    if eval_parameters is None:
        eval_parameters = {}
    prefix, suffix = model_utils._parse_model_uri(model_uri)

    if prefix == "openai":
        return model_utils._call_openai_api(suffix, payload, eval_parameters)
    elif prefix == "gateway":
        return model_utils._call_gateway_api(suffix, payload, eval_parameters)
    elif prefix == "endpoints":
        return model_utils._call_deployments_api(suffix, payload, eval_parameters)
    elif prefix in ("model", "runs"):
        return _load_model_or_server(model_uri, payload)
    else:
        raise MlflowException(f"Unknown model uri prefix '{prefix}'")


# PATCH: Until new version adds the correct _load_model_or_server function
model_utils.score_model_on_payload = score_model_on_payload


class EnumGenAIEvalDimension(Enum):
    """
    Enum representing the different evaluation dimensions for GenAI models.
    """
    DIMENSION_A_B_TESTING = "A_B_testing"
    DIMENSION_BIAS = "Bias"
    DIMENSION_CORRECTNESS = "Correctness"
    DIMENSION_HUMAN_FEEDBACK = "Human Feedback"
    DIMENSION_LATENCY = "Latency"
    DIMENSION_PERFORMANCE = "Performance"
    DIMENSION_RAG = "RAG"
    DIMENSION_ROBUSTNESS = "Robustness"
    DIMENSION_SAFETY_MONITORING = "Safety Monitoring"


class EnumGenAIEvalMetric(Enum):
    """
    Enum representing the different evaluation metrics for GenAI models.
    """
    METRIC_BERT_SCORE = "bert-score"
    METRIC_LLM_JUDGE_BIAS = (
        "llm-judge-bias"  # Like toxicity metric, but with LLM as judge
    )
    METRIC_LLM_JUDGE_COHERENCE = "llm-judge-answer_correctness"
    METRIC_LLM_JUDGE_SIMILARITY = "llm-judge-answer_similarity"
    METRIC_LLM_JUDGE_FAITHFULNESS = "llm-judge-faithfulness"  # Require: context
    METRIC_LLM_JUDGE_ANSWER_RELEVANCE = "llm-judge-answer_relevance"
    METRIC_LLM_JUDGE_RELEVANCE = "llm-judge-relevance"  # Require: context
    METRIC_ARI_GRADE_LEVEL = "ari_grade_level"
    METRIC_FLESCH_KINCAID_GRADE_LEVEL = "flesch_kincaid_grade_level"
    METRIC_LATENCY = "latency"
    METRIC_ROUGE_2 = "rouge2"
    METRIC_ROUGE_L = "rougeL"
    METRIC_ROUGE_L_sum = "rougeLsum"
    METRIC_TOXICITY = "toxicity"
    METRIC_RAG_PRECISION_AT_K = "rag_precision_at_k"
    METRIC_RAG_RECALL_AT_K = "rag_recall_at_k"
    METRIC_RAG_NDCG_AT_K = "rag_ndcg_at_k"


class CustomFoundationalModelURI(mlflow.pyfunc.PythonModel):
    """
    This class is a subclass of mlflow.pyfunc.PythonModel and is used to create a custom foundational model URI.

    Attributes:
        path_local_logs (str): The path to the local logs file.

    Methods:
        load_context(self, context):
            Loads the context for the custom foundational model.
        predict(self, context, model_input):
            Predicts the output based on the input.
    """
    def __init__(self, path_local_logs="./logs.txt"):
        super().__init__()
        self.path_local_logs = path_local_logs

    def load_context(self, context):
        """
        Loads the context for the custom foundational model.

        Args:
            context (object): The context object.

        Returns:
            None
        """
        # TODO: use artifacts or similar to load models
        # model_path = context.artifacts['my_model_path']
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_NOTHING
        verify_ssl = False

        gen_ai_llm = GenAILLM(
            platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
            model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_INSTANT,
            parameters_inference={
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 0.8,
                "stop_sequences": ["User:"],
                "max_tokens_to_sample": 128,
                "top_k": 5,
            },
            platform_configuration={
                # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "aws_access_key_id": os.environ["aws_access_key_id"],
                "aws_secret_access_key": os.environ["aws_secret_access_key"],
                "region_name": os.environ["region_name"],
            },
            verify_ssl=verify_ssl,
            verbose_level=verbose_level,
            metadata=pd.read_csv(f"{os.getcwd()}/model_prices.csv", delimiter=";"),
        )

        self.gen_ai_memory = GenAIMemory()

        self.node = GenAINodeChain(
            "llm_judge_chain",
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
                """,
                partials={},
            ),
            gen_ai_output_parser=None,
            func_invoke=None,
            verbose_level=verbose_level,
        )

    def predict(self, context, model_input):
        """
        Performs a prediction using the custom foundational model.

        Args:
            context (dict): The context information for the prediction.
            model_input (dict): The input data for the prediction.

        Returns:
            The result of the prediction.
        """
        # Directly return the square of the input
        node_result = self.node.invoke(
            {"system_prompt": model_input}, self.gen_ai_memory
        )[self.node.output_key]

        # TODO: Centralize this types of logs
        with open(self.path_local_logs, "a") as f_in:
            f_in.write(str(self.node))

        return node_result


class GenAIEval:
    """
    A class for evaluating the performance of a generative AI model.

    Attributes:
        map_enum_to_text_metric (dict): A dictionary that maps enumeration values to corresponding text metric functions.
        verbose_level (EnumLogs): The verbosity level for logging.
        verbose (bool): A flag indicating whether to enable verbose logging.
        language (str): The language of the text being evaluated.

    Methods:
        __init__(self, verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO, language="es"):
            Initializes the GenAIEval object with the specified verbosity level and language.

        evaluate(self, eval_df: pd.DataFrame, metrics: List[EnumGenAIEvalMetric], run_name="common-eval", judge_model=None):
            Evaluates the performance of a generative AI model using the specified metrics and DataFrame.
    """

    map_enum_to_text_metric = {
        EnumGenAIEvalMetric.METRIC_ARI_GRADE_LEVEL.value: ari_grade_level(),
        EnumGenAIEvalMetric.METRIC_FLESCH_KINCAID_GRADE_LEVEL.value: flesch_kincaid_grade_level(),
        EnumGenAIEvalMetric.METRIC_LATENCY.value: latency(),
        EnumGenAIEvalMetric.METRIC_ROUGE_2.value: rouge2(),
        EnumGenAIEvalMetric.METRIC_ROUGE_L.value: rougeL(),
        EnumGenAIEvalMetric.METRIC_ROUGE_L_sum.value: rougeLsum(),
        # EnumGenAIEvalMetric.METRIC_TOXICITY.value: toxicity(),
    }

    def __init__(
        self, verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO, language="es"
    ):
        """
        Initializes the GenAIEval object with the specified verbosity level and language.

        Args:
            verbose_level (EnumLogs): The verbosity level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.
            language (str): The language of the text being evaluated. Defaults to "es".
        """
        self.verbose_level = verbose_level
        self.verbose = self.verbose_level.value >= 1
        self.language = language

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        metrics: List[EnumGenAIEvalMetric],
        run_name="common-eval",
        judge_model=None,
    ):
        """
        Evaluates the performance of a generative AI model using the specified metrics and DataFrame.

        Args:
            eval_df (pd.DataFrame): The DataFrame containing the evaluation data.
            metrics (List[EnumGenAIEvalMetric]): The list of metrics to be used for evaluation.
            run_name (str): The name of the MLflow run. Defaults to "common-eval".
            judge_model (optional): The model used for judging the performance of the generative AI model.

        Returns:
            pd.DataFrame: The updated DataFrame with the evaluation results.
        """

        mlflow_metrics = []
        other_metrics = []

        for metric in metrics:
            if metric in (
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_ANSWER_RELEVANCE,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_BIAS,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_COHERENCE,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_FAITHFULNESS,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_RELEVANCE,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_SIMILARITY,
            ):
                if judge_model is not None:
                    # TODO: add more examples for llm judge
                    if metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_ANSWER_RELEVANCE:
                        mlflow_metrics.append(answer_relevance(model=judge_model))

                    elif metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_BIAS:
                        raise NotImplementedError()

                    elif metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_COHERENCE:
                        mlflow_metrics.append(answer_correctness(model=judge_model))

                    elif metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_FAITHFULNESS:
                        mlflow_metrics.append(faithfulness(model=judge_model))

                    elif metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_RELEVANCE:
                        mlflow_metrics.append(relevance(model=judge_model))

                    elif metric == EnumGenAIEvalMetric.METRIC_LLM_JUDGE_SIMILARITY:
                        mlflow_metrics.append(answer_similarity(model=judge_model))
                else:
                    Logging.log(
                        f"Judge model is None, can't put: {metric.value}",
                        EnumLogs.LOG_LEVEL_WARNING,
                    )

            elif metric in (
                EnumGenAIEvalMetric.METRIC_ARI_GRADE_LEVEL,
                EnumGenAIEvalMetric.METRIC_FLESCH_KINCAID_GRADE_LEVEL,
                EnumGenAIEvalMetric.METRIC_LATENCY,
                EnumGenAIEvalMetric.METRIC_ROUGE_2,
                EnumGenAIEvalMetric.METRIC_ROUGE_L,
                EnumGenAIEvalMetric.METRIC_ROUGE_L_sum,
                EnumGenAIEvalMetric.METRIC_TOXICITY,
            ):
                map_metric = GenAIEval.map_enum_to_text_metric.get(metric.value, None)
                if map_metric is not None:
                    mlflow_metrics.append(map_metric)
                else:
                    Logging.log(
                        f"Not implemented, can't put: {metric}",
                        EnumLogs.LOG_LEVEL_ERROR,
                    )
                    raise NotImplementedError()
            else:
                other_metrics.append(metric)

        if len(mlflow_metrics) > 0:

            def model(input_df):
                answers = []
                for _, row in input_df.iterrows():
                    # TODO: allow real time prediction instead of predicting before
                    # answer = chain.invoke({"question": row["question"], "context": row["context"]})
                    answer = row["predictions"]
                    answers.append(answer)

                return answers

            with mlflow.start_run(run_name=run_name):
                self.results = mlflow.evaluate(
                    model,
                    eval_df,
                    # model_type="question-answering",
                    # evaluators="default",
                    targets="targets",
                    extra_metrics=mlflow_metrics,
                    # evaluator_config={
                    #     "col_mapping": {
                    #         "inputs": "question",
                    #     }
                    # },
                )
                eval_df = self.results.tables["eval_results_table"]

        for metric in other_metrics:

            if metric == EnumGenAIEvalMetric.METRIC_BERT_SCORE:

                P, R, F1 = score(
                    list(eval_df["predictions"].values),
                    list(eval_df["inputs"].values),
                    lang=self.language,
                    verbose=self.verbose,
                )
                eval_df[f"{EnumGenAIEvalMetric.METRIC_BERT_SCORE.value}_P"] = list(
                    map(float, P)
                )
                eval_df[f"{EnumGenAIEvalMetric.METRIC_BERT_SCORE.value}_R"] = list(
                    map(float, R)
                )
                eval_df[f"{EnumGenAIEvalMetric.METRIC_BERT_SCORE.value}_F1"] = list(
                    map(float, F1)
                )
                # TODO: log params and other metrics outside mlflow
            else:
                if "rag" in metric.value.lower():
                    Logging.log(
                        f"If you want to evaluate rag, use 'evaluate_rag' not 'evaluate'. We can't put: {metric.value}",
                        EnumLogs.LOG_LEVEL_WARNING,
                    )
                else:
                    Logging.log(
                        f"Unknown metric, can't put: {metric.value}",
                        EnumLogs.LOG_LEVEL_WARNING,
                    )

        return eval_df

    def evaluate_rag(
        self,
        eval_df: pd.DataFrame,
        metrics: List[EnumGenAIEvalMetric],
        run_name="common-rag-eval",
        ks=[],
    ):

        assert len(metrics) > 0
        mlflow_metrics = []

        if len(ks) == 0:
            ks = [1]  # at 1

        for metric in metrics:

            if metric in (
                EnumGenAIEvalMetric.METRIC_RAG_PRECISION_AT_K,
                EnumGenAIEvalMetric.METRIC_RAG_RECALL_AT_K,
                EnumGenAIEvalMetric.METRIC_RAG_NDCG_AT_K,
            ):

                metric_func = None
                if metric == EnumGenAIEvalMetric.METRIC_RAG_PRECISION_AT_K:
                    metric_func = precision_at_k
                elif metric == EnumGenAIEvalMetric.METRIC_RAG_RECALL_AT_K:
                    metric_func = recall_at_k
                elif metric == EnumGenAIEvalMetric.METRIC_RAG_NDCG_AT_K:
                    metric_func = ndcg_at_k
                else:
                    Logging.log(
                        f"Metric invalid for RAG: {metric}", EnumLogs.LOG_LEVEL_ERROR
                    )
                    raise NotImplementedError()

                for k in ks:
                    mlflow_metrics.append(metric_func(k))

        if len(mlflow_metrics) > 0:

            with mlflow.start_run(run_name=run_name):

                self.results = mlflow.evaluate(
                    data=eval_df,
                    # model_type="retriever",
                    targets="ground_truth_context",
                    predictions="retrieved_context",
                    # evaluators="default",
                    extra_metrics=mlflow_metrics,
                )

        return self.results.tables["eval_results_table"]
