<!-- Bilingual Links -->
English | [中文 / Chinese](README-zh.md)

# ComfyUI FlexAI Plugin

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/your-repo/Comfyui-flexai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modern, unified ComfyUI plugin for OpenAI-compatible APIs with enhanced debugging and dual-mode image processing capabilities.

I wrote this plugin mainly to simplify my own workflow, allowing access to LLM or image generation through standardized OpenAI interfaces. Recently, I've been mainly using this node to play with Gemini-2.5-Flash-Image generation (haven't tested the official Gemini interface, I'm connecting to an OpenAI-compatible endpoint), and the results are quite good.

## Recent Updates
- 2025-9-2 v1.0 release, tested integration with [tuzi](https://api.tu-zi.com/) and [GB](https://github.com/snailyp/gemini-balance) works perfectly
- 2025-8-31 Plugin initialization

## ✨ Key Features

### 🌐 Multiple API Sources Support
- **Flexible Configuration**: Support multiple API providers via `.env` file, so if you have multiple token channels you don't have to switch back and forth, they can be shared.
- **Dynamic Selection**: Switch between providers without restarting ComfyUI

### 🖼️ OpenAI Image Node (`flexai:openai_image`)
![](thumb/flexai-image-node.jpg)
**Dual-Mode Operation:**
- **Edit Mode**: Provide 1-4 images → Uses `images.edit` API
- **Generate Mode**: No images → Uses `images.generate` API

### 💬 OpenAI Text Node (`flexai:openai_text`)
![](thumb/flexai-text-node.jpg)
**Multimodal Text Generation:**
- Pure text or vision-language understanding
- Support 1-4 reference images with auto-scaling
- Streaming and non-streaming modes
- **Debug Mode**: Complete JSON logging for all operations

## Quick Start

### Installation

1. Clone to ComfyUI custom nodes:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/Comfyui-flexai.git
   ```

2. Install dependencies:
   ```bash
   cd Comfyui-flexai
   pip install -r requirements.txt
   ```

3. Configure providers (see Configuration section)
4. Restart ComfyUI

## Configuration

Create a `.env` file in the plugin root directory.

### Single Provider
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional
```

### Multiple Providers (Recommended)
```bash
# Define provider list
OPENAI_PROVIDERS=openai,anthropic,custom

# OpenAI
OPENAI_API_KEY_openai=sk-your-openai-key
OPENAI_API_BASE_openai=https://api.openai.com/v1

# Anthropic (via OpenAI-compatible endpoint)
OPENAI_API_KEY_anthropic=sk-your-anthropic-key  
OPENAI_API_BASE_anthropic=https://api.anthropic.com/v1

# Custom endpoint
OPENAI_API_KEY_custom=your-custom-key
OPENAI_API_BASE_custom=https://your-api.example.com/v1
```

## Node Parameters

### Image Node (`flexai:openai_image`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `dall-e-3`, `dall-e-2`) |
| `prompt` | String | Generation/editing prompt |
| `image_1-4` | Image | Optional images (edit mode if any provided) |
| `size` | String | Output size (e.g., `1024x1024`) |
| `debug` | Boolean | **Enable detailed debug logging** |

### Text Node (`flexai:openai_text`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `gpt-4o`, `gpt-3.5-turbo`) |
| `system_prompt` | String | System message |
| `user_prompt` | String | User message |
| `image_1-4` | Image | Optional reference images |
| `max_tokens` | Integer | Maximum response tokens |
| `temperature` | Float | Sampling temperature (0.0-1.0) |
| `stream` | Boolean | Enable streaming mode |
| `debug` | Boolean | **Enable detailed debug logging** |

---

## Example Workflows

### Product Placement
![Product Placement Workflow Thumbnail](workflows/flexai-product-placement.jpg)
[Download Workflow](workflows/flexai-product-placement.json)

### Figure Redress
![Figure Redress Workflow Thumbnail](workflows/flexai-figure-redress.jpg)
[Download Workflow](workflows/flexai-figure-redress.json)

---

Built with ❤️ for the ComfyUI community
