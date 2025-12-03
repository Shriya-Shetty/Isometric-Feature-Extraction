# Isometric-Feature-Extraction
Isometric Feature Extraction is a computer-vision method that automatically detects and extracts pipes, fittings, supports, symbols, and annotations from 2D isometric engineering drawings and converts them into structured digital information. This project extracts support-related data from piping isometric drawings using Google's Gemini models.

## Features
- Automatic extraction of support tags (S1, S2, S3, ...)
- Parses annotations (P-4P-SPS-xxxxx)
- Outputs structured CSVs
- PDF-aware LLM processing
- Configurable prompt-based extraction

## Usage

### 1. Add your API key
Create a `.env` file:
