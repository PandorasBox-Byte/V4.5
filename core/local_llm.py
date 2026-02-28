#!/usr/bin/env python3
"""Local LLM Integration - Ollama and llama.cpp support.

Provides local inference with fallback chain:
  ollama → llama.cpp → fine-tuned GPT-2 → cloud backend

Environment Variables:
  EVOAI_LOCAL_LLM_PROVIDER: "ollama" | "llamacpp" | "auto" (default: auto)
  EVOAI_LOCAL_MODEL: Model name (default: "llama3.2:1b" for ollama)
  EVOAI_LLM_PRIORITY: "local" | "cloud" (default: local)
  EVOAI_LLAMACPP_PATH: Path to llama-cli binary (default: auto-detect)
  EVOAI_LOCAL_LLM_TIMEOUT: Inference timeout in seconds (default: 30)
"""

import os
import sys
import subprocess
import shutil
import json
from typing import Optional, Dict, Any
from pathlib import Path


class LocalLLM:
    """Manages local LLM inference via ollama or llama.cpp."""
    
    def __init__(self):
        self.provider = os.environ.get("EVOAI_LOCAL_LLM_PROVIDER", "auto").lower()
        self.model = os.environ.get("EVOAI_LOCAL_MODEL", "llama3.2:1b")
        self.timeout = int(os.environ.get("EVOAI_LOCAL_LLM_TIMEOUT", "30"))
        self.priority = os.environ.get("EVOAI_LLM_PRIORITY", "local").lower()
        
        # Detect available providers
        self.ollama_available = self._check_ollama()
        self.llamacpp_available = self._check_llamacpp()
        
        # Select provider based on availability and config
        if self.provider == "auto":
            if self.ollama_available:
                self.active_provider = "ollama"
            elif self.llamacpp_available:
                self.active_provider = "llamacpp"
            else:
                self.active_provider = None
        else:
            self.active_provider = self.provider if self._provider_available(self.provider) else None
    
    def _check_ollama(self) -> bool:
        """Check if ollama is installed and running."""
        if not shutil.which("ollama"):
            return False
        
        try:
            # Check if ollama service is responsive
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5,
                text=True
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_llamacpp(self) -> bool:
        """Check if llama.cpp is available."""
        llamacpp_path = os.environ.get("EVOAI_LLAMACPP_PATH")
        if llamacpp_path and os.path.exists(llamacpp_path):
            return True
        
        # Check common binary names
        for binary in ["llama-cli", "llama", "main"]:
            if shutil.which(binary):
                return True
        
        return False
    
    def _provider_available(self, provider: str) -> bool:
        """Check if specific provider is available."""
        if provider == "ollama":
            return self.ollama_available
        elif provider == "llamacpp":
            return self.llamacpp_available
        return False
    
    def is_available(self) -> bool:
        """Check if any local LLM provider is available."""
        return self.active_provider is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current provider status."""
        return {
            "active_provider": self.active_provider,
            "ollama_available": self.ollama_available,
            "llamacpp_available": self.llamacpp_available,
            "model": self.model,
            "priority": self.priority
        }
    
    def list_ollama_models(self) -> list:
        """List available ollama models."""
        if not self.ollama_available:
            return []
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode != 0:
                return []
            
            # Parse ollama list output
            models = []
            for line in result.stdout.splitlines()[1:]:  # Skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        except Exception:
            return []
    
    def pull_ollama_model(self, model_name: str) -> bool:
        """Pull an ollama model if not present."""
        if not self.ollama_available:
            return False
        
        try:
            print(f"Pulling ollama model: {model_name} (this may take a while)...")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                timeout=600,  # 10 min timeout for model download
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False
    
    def generate_ollama(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Generate text using ollama."""
        if not self.ollama_available:
            return None
        
        # Check if model is available
        available_models = self.list_ollama_models()
        if self.model not in available_models:
            print(f"Model {self.model} not found locally. Attempting to pull...")
            if not self.pull_ollama_model(self.model):
                return None
        
        try:
            # Use JSON mode for structured output
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                timeout=self.timeout,
                text=True,
                input=""
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except subprocess.TimeoutExpired:
            print(f"Ollama generation timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"Ollama generation error: {e}")
            return None
    
    def generate_llamacpp(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Generate text using llama.cpp."""
        if not self.llamacpp_available:
            return None
        
        # Find llama.cpp binary
        llamacpp_path = os.environ.get("EVOAI_LLAMACPP_PATH")
        if not llamacpp_path:
            for binary in ["llama-cli", "llama", "main"]:
                path = shutil.which(binary)
                if path:
                    llamacpp_path = path
                    break
        
        if not llamacpp_path:
            return None
        
        # Note: llama.cpp requires a model file path, not a model name
        # This is a simplified implementation - real usage would need model path config
        # For now, return None to indicate llamacpp needs additional setup
        print("llama.cpp requires model file path configuration (EVOAI_LLAMACPP_MODEL_PATH)")
        return None
    
    def generate(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Generate text using active local LLM provider."""
        if not self.is_available():
            return None
        
        if self.active_provider == "ollama":
            return self.generate_ollama(prompt, max_tokens)
        elif self.active_provider == "llamacpp":
            return self.generate_llamacpp(prompt, max_tokens)
        
        return None
    
    def install_guide(self) -> str:
        """Return installation guide for local LLM providers."""
        guide = []
        
        if not self.ollama_available:
            guide.append("Ollama:")
            guide.append("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
            guide.append("  Then: ollama pull llama3.2:1b")
        
        if not self.llamacpp_available:
            guide.append("\nllama.cpp:")
            guide.append("  Build from source: https://github.com/ggerganov/llama.cpp")
            guide.append("  Or use pre-built binaries for your platform")
        
        return "\n".join(guide) if guide else "Local LLM providers are available!"


# Singleton instance
_local_llm = None


def get_local_llm() -> LocalLLM:
    """Get or create singleton LocalLLM instance."""
    global _local_llm
    if _local_llm is None:
        _local_llm = LocalLLM()
    return _local_llm


if __name__ == "__main__":
    # Test local LLM availability
    llm = get_local_llm()
    status = llm.get_status()
    
    print("Local LLM Status:")
    print(f"  Active Provider: {status['active_provider'] or 'None'}")
    print(f"  Ollama Available: {status['ollama_available']}")
    print(f"  llama.cpp Available: {status['llamacpp_available']}")
    print(f"  Model: {status['model']}")
    print(f"  Priority: {status['priority']}")
    
    if llm.is_available():
        print("\nTesting generation...")
        response = llm.generate("What is 2+2?", max_tokens=50)
        print(f"Response: {response}")
    else:
        print("\nNo local LLM available. Installation guide:")
        print(llm.install_guide())
