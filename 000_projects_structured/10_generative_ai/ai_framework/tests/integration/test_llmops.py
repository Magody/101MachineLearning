from ai_framework.GenAIEval import EnumGenAIEvalMetric, GenAIEval
from ai_framework.LoggingAndTelemetry import EnumLogs
import pandas as pd
from ai_framework.test.utils import no_ssl_verification, get_mode
import mlflow
import sys

mode = get_mode(sys.argv)


def rag_evaluation():

    data_rag = pd.DataFrame(
        {
            "questions": [
                "What is MLflow?",
                "What is Databricks?",
                "How to serve a model on Databricks?",
                "How to enable MLflow Autologging for my workspace by default?",
            ],
            "retrieved_context": [
                [
                    "mlflow/index.html",
                    "mlflow/quick-start.html",
                ],
                [
                    "introduction/index.html",
                    "getting-started/overview.html",
                ],
                [
                    "machine-learning/model-serving/index.html",
                    "machine-learning/model-serving/model-serving-intro.html",
                ],
                [],
            ],
            "ground_truth_context": [
                ["mlflow/index.html"],
                ["introduction/index.html"],
                [
                    "machine-learning/model-serving/index.html",
                    "machine-learning/model-serving/llm-optimized-model-serving.html",
                ],
                ["mlflow/databricks-autologging.html"],
            ],
        }
    )

    gen_ai_eval = GenAIEval(EnumLogs.LOG_LEVEL_DEBUG, language="es")
    evaluation_rag = gen_ai_eval.evaluate_rag(
        data_rag,
        metrics=[
            EnumGenAIEvalMetric.METRIC_RAG_PRECISION_AT_K,
            EnumGenAIEvalMetric.METRIC_RAG_RECALL_AT_K,
            EnumGenAIEvalMetric.METRIC_RAG_NDCG_AT_K,
        ],
        ks=[1, 2],
    )
    return evaluation_rag


def test_rag_evaluation():
    if mode == "debug":
        pass
    elif mode in ("test", "prod"):
        evaluation_rag = rag_evaluation()
        assert "precision_at_1/score" in evaluation_rag.columns
        assert "recall_at_1/score" in evaluation_rag.columns
        assert "ndcg_at_1/score" in evaluation_rag.columns
        assert "precision_at_2/score" in evaluation_rag.columns
        assert "recall_at_2/score" in evaluation_rag.columns
        assert "ndcg_at_2/score" in evaluation_rag.columns


def metrics_mlflow():

    with no_ssl_verification():

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("metrics-evaluation")

        """
        # runs:/414a9338ccb94d4eb54f11a5168f1dc2/mymodel-01
        custom_foundational_model = CustomFoundationalModelURI()
        # Save the model
        with mlflow.start_run(run_name="custom-model"):

            model_info = mlflow.pyfunc.log_model(
                artifact_path="mymodel-01",
                python_model=custom_foundational_model
            )
            model_uri = model_info.model_uri

        loaded_model = mlflow.pyfunc.load_model(
            model_uri=model_info.model_uri
        )
        """
        model_uri = "runs:/414a9338ccb94d4eb54f11a5168f1dc2/mymodel-01"

        judge_model = model_uri
        print(judge_model)

        eval_df = pd.DataFrame(
            {
                "inputs": [
                    "What is MLflow?",
                    "What is Spark?",
                ],
                "context": [
                    "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. "
                    "It was developed by Databricks, a company that specializes in big data and machine learning solutions. "
                    "MLflow is designed to address the challenges that data scientists and machine learning engineers "
                    "face when developing, training, and deploying machine learning models.",
                    "Apache Spark is an open-source, distributed computing system designed for big data processing and "
                    "analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, "
                    "offering improvements in speed and ease of use. Spark provides libraries for various tasks such as "
                    "data ingestion, processing, and analysis through its components like Spark SQL for structured data, "
                    "Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
                ],
                "targets": [
                    "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. "
                    "It was developed by Databricks, a company that specializes in big data and machine learning solutions. "
                    "MLflow is designed to address the challenges that data scientists and machine learning engineers "
                    "face when developing, training, and deploying machine learning models.",
                    "Apache Spark is an open-source, distributed computing system designed for big data processing and "
                    "analytics. It was developed in response to limitations of the Hadoop MapReduce computing model, "
                    "offering improvements in speed and ease of use. Spark provides libraries for various tasks such as "
                    "data ingestion, processing, and analysis through its components like Spark SQL for structured data, "
                    "Spark Streaming for real-time data processing, and MLlib for machine learning tasks",
                ],
                "predictions": [
                    "MLflow is an open-source platform that provides handy tools to manage Machine Learning workflow lifecycle in a simple way",
                    "Spark is a popular open-source distributed computing system designed for big data processing and analytics.",
                ],
            }
        )

        gen_ai_eval = GenAIEval(EnumLogs.LOG_LEVEL_DEBUG)
        evaluation = gen_ai_eval.evaluate(
            eval_df,
            metrics=[
                EnumGenAIEvalMetric.METRIC_BERT_SCORE,
                # EnumGenAIEvalMetric.METRIC_LLM_JUDGE_BIAS,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_COHERENCE,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_SIMILARITY,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_FAITHFULNESS,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_ANSWER_RELEVANCE,
                EnumGenAIEvalMetric.METRIC_LLM_JUDGE_RELEVANCE,
                EnumGenAIEvalMetric.METRIC_ARI_GRADE_LEVEL,
                EnumGenAIEvalMetric.METRIC_FLESCH_KINCAID_GRADE_LEVEL,
                # EnumGenAIEvalMetric.METRIC_LATENCY,
                EnumGenAIEvalMetric.METRIC_ROUGE_2,
                EnumGenAIEvalMetric.METRIC_ROUGE_L,
                EnumGenAIEvalMetric.METRIC_ROUGE_L_sum,
                # EnumGenAIEvalMetric.METRIC_TOXICITY
            ],
            judge_model=judge_model,
        )
        return evaluation

    return None


def test_metrics_mlflow():
    if mode == "debug":
        pass
    elif mode in ("test", "prod"):
        columns_reference = set(
            [
                "answer_correctness/v1/score",
                "answer_similarity/v1/score",
                "faithfulness/v1/score",
                "answer_relevance/v1/score",
                "relevance/v1/score",
                "ari_grade_level/v1/score",
                "flesch_kincaid_grade_level/v1/score",
                "rouge2/v1/score",
                "rougeL/v1/score",
                "rougeLsum/v1/score",
                "bert-score_P",
                "bert-score_R",
                "bert-score_F1",
            ]
        )

        evaluation = metrics_mlflow()

        assert len(set(evaluation.columns).intersection(columns_reference)) == len(
            columns_reference
        )


if mode in ("test", "prod"):
    test_rag_evaluation()
