# assistant_surete_nucleaire/generation/generator.py
import ollama
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.prompts.prompt_loader import PromptLoader

class Generator:
    def __init__(self, prompt_version: str = None):
        self.model = config.generation.local_model_generation
        self.max_tokens = config.generation.max_tokens
        self.temperature = config.generation.temperature
        self.prompt_loader = PromptLoader()

    def generate(self, question: str, chunks: list) -> str:
        try:
            # Construction du contexte
            context_texts = []
            for item in chunks:
                if isinstance(item, tuple):
                    chunk = item[0]
                else:
                    chunk = item
                context_texts.append(f"[{chunk.doc_title} p.{chunk.page}] {chunk.text}")
            context = "\n\n---\n\n".join(context_texts)

            messages = self.prompt_loader.render(
                "generation",
                context=context,
                question=question
            )

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": messages["system"]},
                    {"role": "user", "content": messages["user"]}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Erreur lors de la génération : {str(e)}"