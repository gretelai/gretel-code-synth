"""
Basic usage example for gretel-code-synth.
Shows how to process a repository and generate training examples.
"""

import logging
from pathlib import Path

from gretel_code_synth import CodeSynth

logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize CodeSynth
        synth = CodeSynth(gretel_api_key="prompt")

        # Process repository
        synth.process_repository(
            repo_url="https://github.com/gretelai/gretel-blueprints.git",
            days=30,
            max_batch_size=5,
            min_batch_size=2,
        )

        # Configure and generate dataset
        synth.configure()

        # Generate preview
        logger.info("Generating preview...")
        preview = synth.designer.generate_dataset_preview()
        preview.display_sample_record()

        output_path = Path("generated_preview.jsonl")
        preview.dataset.to_json(output_path, orient="records", lines=True)

        # Generate full dataset
        # logger.info("\nGenerating full dataset...")
        # dataset = synth.designer.generate_dataset(num_records=10)  # Adjust number as needed

        # Save to disk
        # output_path = Path("generated_data.csv")
        # dataset.to_csv(output_path, index=False)
        # logger.info(f"\nSaved dataset to: {output_path}")

        # Print summary stats
        # logger.info(f"\nGenerated {len(dataset)} records with columns:")
        # for col in dataset.columns:
        #    logger.info(f"- {col}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
