# assistant_surete_nucleaire/generation/generator.py
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
        # Construction du contexte (inchangé)
        context_texts = []
        for item in chunks:
            if isinstance(item, tuple):
                chunk = item[0]
            else:
                chunk = item
            context_texts.append(f"[{chunk.doc_title} p.{chunk.page}] {chunk.text}")
        context = "\n\n---\n\n".join(context_texts)

        # Rendre les prompts depuis le YAML
        messages = self.prompt_loader.render(
            "generation",
            context=context,
            question=question
        )

        # Ollama attend une structure "system" et "user" (comme OpenAI)
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


    def generate_hypothetical(self, question: str) -> str:
        """Génération d'une réponse hypothétique (HyDE) sans contexte."""
        messages = self.prompt_loader.render("hyde", question=question)

        # On utilise un peu plus de créativité pour HyDE (0.3) et un max_tokens un peu plus court
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]}
            ],
            options={
                "temperature": 0.3,
                "num_predict": 300,  # Pas besoin d'un énorme paragraphe pour le retrieval
            }
        )
        return response["message"]["content"].strip()