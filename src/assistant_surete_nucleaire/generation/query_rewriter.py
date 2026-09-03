# assistant_surete_nucleaire/generation/query_rewriter.py
import ollama
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.prompts.prompt_loader import PromptLoader

class QueryRewriter:
    def __init__(self):
        self.model = config.generation.local_model_generation
        self.prompt_loader = PromptLoader()

    def rewrite(self, question: str, history: list[dict]) -> str:
        if not history:
            return question

        # Formater l'historique
        history_text = "\n".join([
            f"{msg['role'].capitalize()} : {msg['content']}"
            for msg in history
        ])

        messages = self.prompt_loader.render(
            "query_rewrite",
            history=history_text,
            question=question
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 256,
            }
        )
        return response["message"]["content"].strip()