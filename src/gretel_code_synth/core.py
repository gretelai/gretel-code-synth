import logging
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import git
import nbformat
from gretel_client.navigator import DataDesigner
from gretel_client.navigator.tasks.types import CategoricalDataSeeds

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

        # Default prompts aligned with library usage patterns
        self.default_prompts = {
            "question": """
            Create a practical developer question about using a library for {pattern_type} in a {context_scope} context.
            The example should be at {complexity_level} level and demonstrate {aspects}.
            
            Reference implementation:
            ```python
            {code_examples}
            ```
            
            Focus on {use_case} scenarios.
            
            Requirements:
            - Frame as a specific, actionable use case
            - Include clear requirements and constraints
            - Specify expected behavior and outputs
            - Highlight integration points and dependencies
            - Include relevant configuration or setup needs
            """,
            "code_solution": """
            Write a solution for: {question}
            
            Requirements:
            1. Follow {aspects} principles
            2. Match {complexity_level} level expectations
            3. Include imports and setup
            4. Add comprehensive docstrings with:
               - Purpose and context
               - Parameters
               - Return values
               - Example usage
               - Common pitfalls
            5. Add type hints
            6. Include error handling
            7. Add logging where appropriate
            8. Include example usage
            
            Reference implementation:
            ```python
            {code_examples}
            ```
            
            Make it production-ready and well-documented.
            """,
            "explanation": """
            Explain the solution for: {question}
            
            Focus on:
            1. Overall approach and design choices
            2. Library initialization and setup
            3. Key integration points
            4. Error handling strategy
            5. Resource management
            6. Performance considerations
            7. Best practices demonstrated ({aspects})
            8. Common pitfalls to avoid
            9. Testing approaches
            
            Target this explanation for {complexity_level} level developers.
            Include both "why" and "how" explanations.
            """,
        }


import logging
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import git
import nbformat
from gretel_client.navigator import DataDesigner
from gretel_client.navigator.tasks.types import CategoricalDataSeeds

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

        # Default prompts aligned with library usage patterns
        self.default_prompts = {
            "question": """
            Create a practical developer question about using a library for {pattern_type} in a {context_scope} context.
            The example should be at {complexity_level} level and demonstrate {aspects}.
            
            Reference implementation:
            ```python
            {code_examples}
            ```
            
            Focus on {use_case} scenarios.
            
            Requirements:
            - Frame as a specific, actionable use case
            - Include clear requirements and constraints
            - Specify expected behavior and outputs
            - Highlight integration points and dependencies
            - Include relevant configuration or setup needs
            """,
            "code_solution": """
            Write a solution for: {question}
            
            Requirements:
            1. Follow {aspects} principles
            2. Match {complexity_level} level expectations
            3. Include imports and setup
            4. Add comprehensive docstrings with:
               - Purpose and context
               - Parameters
               - Return values
               - Example usage
               - Common pitfalls
            5. Add type hints
            6. Include error handling
            7. Add logging where appropriate
            8. Include example usage
            
            Reference implementation:
            ```python
            {code_examples}
            ```
            
            Make it production-ready and well-documented.
            """,
            "explanation": """
            Explain the solution for: {question}
            
            Focus on:
            1. Overall approach and design choices
            2. Library initialization and setup
            3. Key integration points
            4. Error handling strategy
            5. Resource management
            6. Performance considerations
            7. Best practices demonstrated ({aspects})
            8. Common pitfalls to avoid
            9. Testing approaches
            
            Target this explanation for {complexity_level} level developers.
            Include both "why" and "how" explanations.
            """,
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

            if not all_examples:
                raise ValueError("No valid code examples found in repository")

            # Store examples for later use
            self.code_examples = dict(all_examples)

        except Exception as e:
            raise Exception(f"Failed to process repository: {e}")
        finally:
            shutil.rmtree(temp_dir)

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
            logger.info("\nIngesting code seeds:")
            for cat, examples in self.code_examples.items():
                logger.info(f"- {cat}: {len(examples)} examples")
            self.designer.ingest_categorical_data_seeds(
                CategoricalDataSeeds(**code_seeds)
            )

            # 2. Ingest task seeds (default or provided)
            task_seeds = (
                seeds
                if seeds is not None
                else {
                    "seed_categories": [
                        {
                            "name": "pattern_type",
                            "values": [
                                "initialization",
                                "basic_usage",
                                "advanced_usage",
                                "error_handling",
                                "optimization",
                                "testing",
                            ],
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
                                            "proper_error_handling",
                                            "clear_documentation",
                                            "type_safety",
                                            "idiomatic_usage",
                                        ],
                                        "anti_pattern": [
                                            "missing_error_handling",
                                            "poor_documentation",
                                            "type_unsafe",
                                            "non_idiomatic",
                                        ],
                                    },
                                }
                            ],
                        },
                    ]
                }
            )
            logger.info("\nIngesting task seeds:")
            for cat in task_seeds["seed_categories"]:
                logger.info(f"- {cat['name']}: {len(cat['values'])} values")
            self.designer.ingest_categorical_data_seeds(
                CategoricalDataSeeds(**task_seeds)
            )

            # 3. Configure prompts to use both code examples and task structure
            generation_prompts = (
                prompts
                if prompts is not None
                else {
                    "question": """
                Create a practical developer question about {pattern_type} at {complexity_level} level.
                The implementation should demonstrate {aspects}.
                
                Use this reference implementation:
                ```python
                {implementations}
                ```
                
                Make it a specific, actionable question that tests understanding of the pattern.
                """,
                    "code_solution": """
                Write a solution for: {question}
                
                Use this reference implementation:
                ```python
                {implementations}
                ```
                
                Requirements:
                - Match {complexity_level} level expectations
                - Follow {aspects} principles
                - Include all necessary imports
                - Add clear docstrings
                - Include error handling
                - Add example usage
                """,
                    "explanation": """
                Explain the solution for: {question}
                
                Focus on:
                1. Overall approach
                2. Key implementation details
                3. How it demonstrates {aspects}
                4. Common pitfalls at {complexity_level} level
                5. Usage examples
                """,
                }
            )

            # Add columns
            for column, prompt in generation_prompts.items():
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
            preview = self.designer.generate_dataset_preview()

            # Log preview details
            logger.info("Preview generated")
            if preview and hasattr(preview, "records") and preview.records:
                logger.info(f"Preview contains {len(preview.records)} records")
                logger.info("Sample record:")
                for key, value in preview.records[0].items():
                    logger.info(f"{key}: {value}")
            else:
                logger.warning("No preview records generated")

        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            raise Exception(f"Failed to generate preview: {e}")

    def _extract_notebook_content(
        self,
        notebook_path: Union[str, Path],
        max_batch_size: int = 3,
        min_batch_size: int = 1,
    ) -> Dict[str, List[str]]:
        """Extract code with smarter batching based on notebook structure."""
        categories = defaultdict(list)
        current_batch = []
        current_context = []

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    current_context.append(cell.source.strip())

                elif cell.cell_type == "code":
                    code = cell.source.strip()
                    if not self._is_valid_code_cell(code):
                        continue

                    if len(current_batch) >= max_batch_size:
                        # Process batch
                        if len(current_batch) >= min_batch_size:
                            code_blocks = [c for c, _ in current_batch]
                            context = [
                                "# " + l
                                for ctx in (ctx for _, ctx in current_batch)
                                for l in ctx
                                if l.strip()
                            ]

                            full_code = "\n".join(code_blocks)
                            category = self._categorize_code_block(full_code)

                            full_block = "\n".join(context + [""] + code_blocks)
                            categories[category].append(full_block)

                        current_batch = []
                        current_context = []

                    current_batch.append((code, current_context))
                    current_context = []

            # Process final batch
            if len(current_batch) >= min_batch_size:
                code_blocks = [c for c, _ in current_batch]
                context = [
                    "# " + l
                    for ctx in (ctx for _, ctx in current_batch)
                    for l in ctx
                    if l.strip()
                ]

                full_code = "\n".join(code_blocks)
                category = self._categorize_code_block(full_code)

                full_block = "\n".join(context + [""] + code_blocks)
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
            preview = self.designer.generate_dataset_preview()

            # Log preview details
            logger.info("Preview generated")
            if preview and hasattr(preview, "records") and preview.records:
                logger.info(f"Preview contains {len(preview.records)} records")
                logger.info("Sample record:")
                for key, value in preview.records[0].items():
                    logger.info(f"{key}: {value}")
            else:
                logger.warning("No preview records generated")

        except Exception as e:
            logger.error(f"Preview generation failed: {str(e)}")
            raise Exception(f"Failed to generate preview: {e}")

    def _extract_notebook_content(
        self,
        notebook_path: Union[str, Path],
        max_batch_size: int = 3,
        min_batch_size: int = 1,
    ) -> Dict[str, List[str]]:
        """Extract code with smarter batching based on notebook structure."""
        categories = defaultdict(list)
        current_batch = []
        current_context = []

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    current_context.append(cell.source.strip())

                elif cell.cell_type == "code":
                    code = cell.source.strip()
                    if not self._is_valid_code_cell(code):
                        continue

                    if len(current_batch) >= max_batch_size:
                        # Process batch
                        if len(current_batch) >= min_batch_size:
                            code_blocks = [c for c, _ in current_batch]
                            context = [
                                "# " + l
                                for ctx in (ctx for _, ctx in current_batch)
                                for l in ctx
                                if l.strip()
                            ]

                            full_code = "\n".join(code_blocks)
                            category = self._categorize_code_block(full_code)

                            full_block = "\n".join(context + [""] + code_blocks)
                            categories[category].append(full_block)

                        current_batch = []
                        current_context = []

                    current_batch.append((code, current_context))
                    current_context = []

            # Process final batch
            if len(current_batch) >= min_batch_size:
                code_blocks = [c for c, _ in current_batch]
                context = [
                    "# " + l
                    for ctx in (ctx for _, ctx in current_batch)
                    for l in ctx
                    if l.strip()
                ]

                full_code = "\n".join(code_blocks)
                category = self._categorize_code_block(full_code)

                full_block = "\n".join(context + [""] + code_blocks)
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
