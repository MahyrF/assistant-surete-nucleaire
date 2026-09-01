# prompt_loader.py
import yaml
import jinja2
from pathlib import Path
from typing import Optional
from assistant_surete_nucleaire.config import config

class PromptLoader:
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = config.paths.base_dir / "prompts"
        self._cache = {}  # Cache pour ne pas relire le YAML à chaque appel

    def load(self, prompt_name: str, version: Optional[str] = None) -> dict:
        """
        Charge le fichier YAML. Si version est None, prend le plus récent
        ou celui spécifié dans config (pour l'instant on prend le fichier par défaut).
        """
        yaml_path = self.prompt_dir / f"{prompt_name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt {prompt_name} non trouvé dans {self.prompt_dir}")

        if yaml_path in self._cache:
            return self._cache[yaml_path]

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._cache[yaml_path] = data
        return data

    def render(self, prompt_name: str, **kwargs) -> dict:
        """
        Rend les templates system et user avec les variables passées.
        Retourne un dict {'system': str, 'user': str} prêt à être envoyé à l'API.
        """
        data = self.load(prompt_name)
        template_env = jinja2.Environment(loader=jinja2.BaseLoader())

        system_template = template_env.from_string(data.get("system", ""))
        user_template = template_env.from_string(data.get("user", ""))

        rendered_system = system_template.render(**kwargs)
        rendered_user = user_template.render(**kwargs)

        return {
            "system": rendered_system,
            "user": rendered_user,
            "version": data.get("version", "unknown")
        }