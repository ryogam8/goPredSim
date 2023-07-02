import sys
import file_utils as fu
from gene_ontology import GeneOntology
from function_prediction import FunctionPrediction
from pathlib import Path
from tqdm import tqdm


def main():
    # read in information
    config_file = sys.argv[1]
    config_data = fu.read_config_file(config_file)
    print(config_data)

    # read in embeddings, annotations, and GO
    embeddings = fu.read_target_embeddings(
        config_data["lookup_ids"], config_data["lookup_targets"]
    )

    go = GeneOntology(config_data["go"])
    go_annotations = fu.read_go_annotations(config_data["annotations"])

    # set ontologies
    if config_data["onto"] == "all":
        ontologies = ["bpo", "mfo", "cco"]
    else:
        ontologies = [config_data["onto"]]

    # set dist cutoffs:
    cutoffs = config_data["thresh"]
    dist_cutoffs = cutoffs.split(",")

    test_embeddings = fu.read_target_embeddings(
        config_data["target_ids"], config_data["targets"]
    )
    for i in tqdm(range(0, len(test_embeddings), 2000)):
        batch_embeddings = dict(
            list(test_embeddings.items())[i : i + 2000]
        )  # 1000要素ずつのバッチを取得

        # perform prediction for each ontology individually
        for o in ontologies:
            predictor = FunctionPrediction(embeddings, go_annotations, go, o)
            predictions_all, _ = predictor.run_prediction_embedding_all(
                batch_embeddings, "cosine", dist_cutoffs, config_data["modus"]
            )

            # write predictions for each distance cutoff
            for dist in predictions_all.keys():
                predictions = predictions_all[dist]
                predictions_out = "{}_{}_{}_{}.txt".format(
                    config_data["output"], dist, o, "cosine"
                )
                FunctionPrediction.write_predictions(predictions, predictions_out)

            # for compatibility with CAFA assessment tool use the following lines
            # team_name = RandomTeam
            # predictions_out = '{}_{}_all_go_{}.txt'.format(team_name, dist, o.upper())
            # FunctionPrediction.write_predictions_cafa(predictions, predictions_out, dist, team_name)


main()
