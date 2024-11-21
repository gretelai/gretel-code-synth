import json
import logging
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import git
import nbformat
from gretel_client.navigator import DataDesigner
from gretel_client.navigator.tasks.types import CategoricalDataSeeds
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSynth:
    """
    Transform GitHub repository notebooks into synthetic code examples for training language models.
    Uses Gretel's Data Designer to generate high-quality, diverse coding examples that demonstrate
    library usage patterns, best practices, and common pitfalls.
    """

    def __init__(
        self,
        gretel_api_key: str,
        endpoint: str = "https://api.gretel.cloud",
        cache: str = "yes",
    ):
        """
        Initialize CodeSynth with Gretel credentials.

        Args:
            gretel_api_key: Your Gretel AI API key
            endpoint: Gretel API endpoint URL
            cache: Whether to cache results ("yes" or "no")
        """
        # Store credentials
        self.api_key = gretel_api_key
        self.endpoint = endpoint
        self.cache = cache

        # Initialize designer
        self.designer = DataDesigner(
            api_key=gretel_api_key, endpoint=endpoint, cache=cache
        )

        self.preview = None

        # Default seed structure optimized for library usage examples
        self.default_seeds = {
            "seed_categories": [
                {
                    "name": "pattern_type",
                    "values": [
                        "initialization",
                        "basic_usage",
                        "advanced_usage",
                        "error_handling",
                        "optimization",
                        "integration",
                        "custom_extensions",
                        "testing",
                    ],
                    "weights": [0.15, 0.25, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05],
                },
                {
                    "name": "complexity_level",
                    "values": ["beginner", "intermediate", "advanced"],
                    "weights": [0.4, 0.4, 0.2],
                },
                {
                    "name": "code_quality",
                    "values": ["exemplar", "anti_pattern"],
                    "weights": [0.9, 0.1],
                    "subcategories": [
                        {
                            "name": "aspects",
                            "values": {
                                "exemplar": [
                                    "clear_initialization",
                                    "proper_error_handling",
                                    "efficient_resource_usage",
                                    "clear_documentation",
                                    "type_safety",
                                    "idiomatic_usage",
                                    "extensible_design",
                                    "comprehensive_testing",
                                ],
                                "anti_pattern": [
                                    "resource_leaks",
                                    "missing_error_handling",
                                    "poor_initialization",
                                    "type_unsafe",
                                    "non_idiomatic",
                                    "brittle_design",
                                    "untested",
                                    "undocumented",
                                ],
                            },
                        }
                    ],
                },
                {
                    "name": "use_case",
                    "values": [
                        "data_processing",
                        "api_integration",
                        "configuration",
                        "persistence",
                        "monitoring",
                        "validation",
                        "transformation",
                        "analysis",
                    ],
                    "weights": [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1],
                },
                {
                    "name": "context_scope",
                    "values": [
                        "standalone_script",
                        "module_component",
                        "service_integration",
                        "notebook_environment",
                        "cli_tool",
                    ],
                    "weights": [0.3, 0.3, 0.2, 0.1, 0.1],
                },
            ]
        }

    def process_repository(
        self,
        repo_url: str,
        days: int = 365,
        max_batch_size: int = 3,
        min_batch_size: int = 1,
    ) -> None:
        """Extract code examples from a GitHub repository's notebooks."""
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(f"Cloning repository: {repo_url}")
            repo = git.Repo.clone_from(repo_url, temp_dir)

            all_examples = defaultdict(list)
            notebooks_total = 0
            notebooks_with_content = 0

            for notebook_path in Path(temp_dir).rglob("*.ipynb"):
                if ".ipynb_checkpoints" in str(notebook_path):
                    continue

                notebooks_total += 1
                try:
                    content = self._extract_notebook_content(
                        notebook_path,
                        max_batch_size=max_batch_size,
                        min_batch_size=min_batch_size,
                    )

                    if content:
                        notebooks_with_content += 1
                        for category, examples in content.items():
                            all_examples[category].extend(examples)

                except Exception as e:
                    logger.error(f"Error processing notebook {notebook_path}: {e}")

            # Print summary statistics
            logger.info("\n=== Repository Processing Summary ===")
            logger.info(f"Notebooks processed: {notebooks_total}")
            logger.info(f"Notebooks with valid content: {notebooks_with_content}")
            logger.info("\nExamples by category:")
            for category, examples in all_examples.items():
                logger.info(f"- {category}: {len(examples)} examples")
            logger.info(
                f"\nTotal examples extracted: {sum(len(examples) for examples in all_examples.values())}"
            )
            with open("seed_examples.jsonl", "w") as f:
                f.write(json.dumps(all_examples, indent=2))

            if not all_examples:
                raise ValueError("No valid code examples found in repository")

            # Store examples for later use
            self.code_examples = dict(all_examples)

            # Preview 3 examples using Rich
            self._preview_code_seeds(all_examples)

        except Exception as e:
            raise Exception(f"Failed to process repository: {e}")
        finally:
            shutil.rmtree(temp_dir)

    def _preview_code_seeds(self, all_examples: Dict[str, List[str]]) -> None:
        """Preview 3 extracted code seeds using Rich."""
        console = Console()
        table = Table(title="Code Seeds Preview")

        table.add_column("Category", style="bold cyan", no_wrap=True)
        table.add_column("Code Example", style="dim", overflow="fold")

        preview_count = 0
        for category, examples in all_examples.items():
            for example in examples[:3]:  # Limit to 3 examples per category
                syntax = Syntax(example, "python", theme="monokai", line_numbers=True)
                table.add_row(category, syntax)
                preview_count += 1
                if preview_count >= 3:
                    break
            if preview_count >= 3:
                break

        console.print(table)

    def configure(
        self, seeds: Optional[dict] = None, prompts: Optional[dict] = None
    ) -> None:
        """Configure dataset generation with custom seeds and prompts."""
        try:
            # Reinitialize designer
            self.designer = DataDesigner(
                api_key=self.api_key, endpoint=self.endpoint, cache=self.cache
            )

            # 1. Ingest code examples
            if not hasattr(self, "code_examples") or not self.code_examples:
                raise ValueError(
                    "No code examples available. Run process_repository first."
                )

            code_seeds = {
                "seed_categories": [
                    {
                        "name": "code_examples",
                        "values": list(self.code_examples.keys()),
                        "subcategories": [
                            {"name": "implementations", "values": self.code_examples}
                        ],
                    }
                ]
            }
            logger.info("🌱 Ingesting code seeds into data design:")
            self.designer.ingest_categorical_data_seeds(
                CategoricalDataSeeds(**code_seeds)
            )

            # 2. Ingest task seeds (default or provided)
            task_seeds = {
                "seed_categories": [
                    {
                        "name": "pattern_type",
                        "values": [
                            "initialization",  # Client setup, basic config
                            "data_operations",  # Working with data/files
                            "error_handling",  # Try/except patterns
                            "resource_mgmt",  # Context managers, cleanup
                            "optimization",  # Performance improvements
                        ],
                        "weights": [0.3, 0.2, 0.2, 0.15, 0.15],
                    },
                    {
                        "name": "complexity_level",
                        "values": ["beginner", "intermediate", "advanced"],
                        "weights": [0.4, 0.4, 0.2],
                    },
                    {
                        "name": "code_aspects",
                        "values": [
                            "proper_error_handling",  # Try/except with logging
                            "clear_documentation",  # Docstrings and comments
                            "resource_management",  # Clean setup/teardown
                            "idiomatic_usage",  # Pythonic patterns
                        ],
                        "weights": [0.3, 0.3, 0.2, 0.2],
                    },
                    {
                        "name": "context_scope",
                        "values": [
                            "standalone_script",  # Single file scripts
                            "library_usage",  # Using as imported lib
                            "service_component",  # Part of larger service
                        ],
                        "weights": [0.4, 0.4, 0.2],
                    },
                ]
            }

            logger.info("\nIngesting task seeds:")
            for cat in task_seeds["seed_categories"]:
                logger.info(f"- {cat['name']}: {len(cat['values'])} values")
            self.designer.ingest_categorical_data_seeds(
                CategoricalDataSeeds(**task_seeds)
            )

            # 3. Configure prompts to use both code examples and task structure
            self.generation_prompts = {
                "question": """
                Create a practical developer question about using {code_examples} in a {context_scope} context, specifically focusing on {pattern_type} patterns.
                The example should be at {complexity_level} level and demonstrate {code_aspects}.
                
                Use this reference implementation as a base:
                ```python
                {implementations}
                ```
                
                Requirements:
                - Frame as a specific use case that builds on the reference implementation
                - Use the same library and API patterns shown
                - Focus on practical scenarios similar to the example
                - Maintain consistency with the library's patterns
                - Specify configuration and setup relevant to this library
                """,
                "code_solution": """
                Write a solution for: {question}
                
                Base your solution on this reference implementation:
                ```python
                {implementations}
                ```
                
                Requirements:
                1. Use the same library and API patterns shown in the reference
                2. Match {complexity_level} level expectations
                3. Demonstrate {code_aspects}
                4. Include necessary imports shown in the reference, including type hints
                5. Add comprehensive docstrings following the library's style
                6. Add type hints consistent with the library's types
                7. Follow the library's error handling patterns
                8. Add appropriate logging following library conventions
                9. Include example usage similar to the reference
                
                Ensure the solution extends or builds upon the reference implementation's patterns.
                """,
                "explanation": """
                Briefly explain how this solution builds on the reference implementation for a {complexity_level} developer:
                1. How it uses the library's patterns
                2. Key implementation details
                3. Important library-specific practices
                4. Things to watch out for when using this library
                """,
            }

            # Add columns
            for column, prompt in self.generation_prompts.items():
                self.designer.add_generated_data_column(
                    name=column,
                    generation_prompt=prompt,
                    llm_type=(
                        "code" if column == "code_solution" else "natural_language"
                    ),
                )

            # Add validators
            self.designer.add_data_validator(
                validator="code", code_lang="python", code_columns=["code_solution"]
            )
            self.designer.add_dataset_evaluation(
                eval_type="text_to_python",
                instruction_column_name="question",
                response_column_name="code_solution",
            )

        except Exception as e:
            logger.error(f"Configuration failed: {str(e)}")
            raise Exception(f"Failed to configure dataset generation: {e}")

    def preview(self) -> None:
        """Generate and display a preview of the dataset."""
        try:
            logger.info("Starting preview generation...")

            # Make sure we've configured the designer
            if not hasattr(self.designer, "_prompts") or not self.designer._prompts:
                logger.info("Designer not configured, running configure() first")
                self.configure()

            # Generate preview
            logger.info("Generating dataset preview...")
            self.preview = self.designer.generate_dataset_preview()
            self.preview.display_sample_record()

        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            raise Exception(f"Failed to generate preview: {e}")

    def _extract_notebook_content(
        self,
        notebook_path: Union[str, Path],
        max_batch_size: int = 3,
        min_batch_size: int = 1,
    ) -> Dict[str, List[str]]:
        """Extract code with context, preserving imports and markdown structure."""
        categories = defaultdict(list)

        def collect_imports(notebook):
            imports = []
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    imports.extend(
                        line.strip()
                        for line in cell.source.split("\n")
                        if line.strip().startswith(("import ", "from "))
                    )
            return list(dict.fromkeys(imports))

        def combine_batch(imports, markdown, code):
            return "\n".join(imports + [""] + markdown + [""] + code).strip()

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            imports = collect_imports(notebook)
            markdown_buffer, current_code = [], []

            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    markdown_buffer.extend(
                        line if line.lstrip().startswith("#") else f"# {line.strip()}"
                        for line in cell.source.split("\n")
                        if line.strip()
                    )

                elif cell.cell_type == "code":
                    code = cell.source.strip()
                    if not code or code.startswith(("!", "%")):
                        continue

                    if all(
                        line.strip().startswith(("import ", "from "))
                        for line in code.split("\n")
                        if line.strip()
                    ):
                        continue

                    current_code.append(code)

                    if len(current_code) >= min_batch_size:
                        full_block = combine_batch(
                            imports, markdown_buffer, current_code
                        )
                        category = self._categorize_code_block(full_block)
                        categories[category].append(full_block)
                        current_code, markdown_buffer = [], []

            if current_code:
                full_block = combine_batch(imports, markdown_buffer, current_code)
                category = self._categorize_code_block(full_block)
                categories[category].append(full_block)

        except Exception as e:
            logger.error(f"Error processing notebook {notebook_path}: {e}")

        return dict(categories)

    def _categorize_code_block(self, code: str) -> str:
        """Determine category for a complete code block."""
        if "import" in code:
            return "initialization"
        if "def " in code or "class " in code:
            return "definitions"
        if "try:" in code:
            return "error_handling"
        if any(viz in code for viz in ["plot", "fig", "ax", "plt"]):
            return "visualization"
        if any(pattern in code for pattern in ["train", "fit", "predict"]):
            return "model_training"
        return "basic_operations"

    def _is_valid_code_cell(self, code: str) -> bool:
        """Check if code cell should be included in batches."""
        # Skip empty, short, or system command cells
        if not code or len(code) < 10 or code.startswith("!") or code.startswith("%"):
            return False

        # Ensure there's actual code content, not just comments
        code_lines = [
            l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")
        ]
        return len(code_lines) > 0
