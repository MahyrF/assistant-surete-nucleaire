# assistant_surete_nucleaire/generation/faithfulness_checker.py
import ollama
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.prompts.prompt_loader import PromptLoader

class FaithfulnessChecker:
    def __init__(self):
        self.model = config.generation.local_model_faithfulness
        self.prompt_loader = PromptLoader()

    def check(self, question: str, answer: str, context_chunks: list) -> dict:
        """
        Retourne : {
            "faithfulness_score": float (0..1),
            "verdicts": list[str],
            "raw_output": str
        }
        """
        # Construire le contexte sous forme de texte
        if context_chunks:
            # context_chunks peut être une liste de chunks (objets Chunk) ou de textes
            if isinstance(context_chunks[0], str):
                context_text = "\n".join(context_chunks)
            else:
                # On suppose que ce sont des objets Chunk avec un attribut .text
                context_text = "\n".join([c.text for c in context_chunks])
        else:
            context_text = "Aucun contexte disponible."

        # Rendre les prompts depuis le YAML
        messages = self.prompt_loader.render(
            "faithfulness",
            question=question,
            context=context_text,
            answer=answer
        )

        # Appel à Ollama avec le petit modèle
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]}
            ],
            options={"temperature": 0.0}  # Strict, sans créativité
        )

        output = response["message"]["content"].strip()

        # Extraction des verdicts
        verdict_lines = [line for line in output.split('\n') if 'OUI' in line or 'NON' in line]

        # Calcul du score : proportion de OUI
        yes_count = sum(1 for line in verdict_lines if 'OUI' in line)
        total_count = len(verdict_lines) if verdict_lines else 1
        score = yes_count / total_count if total_count > 0 else 0.0

        return {
            "faithfulness_score": score,
            "verdicts": verdict_lines,
            "raw_output": output
        }