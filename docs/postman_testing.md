# Common Postman Testing 
Guide to testign local and production hosted web applications


## How to test a POST endpoint with a payload


To test a POST endpoint you need to include a json body along with the request:


```
# URL
http://127.0.0.1:8000/<Endpoint Name>


# Body
{
  "files": [
    "http://arxiv.org/pdf/1706.03762",
    "http://example.com/anotherfile.pdf"
  ]
}
```

## Testing out the dialogue only endpoint  `pdf-to-dialogue-transcript/`

this is the function that is being used:

```python
def generate_only_dialogue(
    files: list,
    openai_api_key: str = None,
    text_model: str = "o1-preview-2024-09-12",
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
    api_base: str = None,
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
    debug = False,
)
```

the json body for postman testing is:

```
{
  "files": [
    "http://arxiv.org/pdf/1706.03762",
    "http://example.com/anotherfile.pdf"
  ]
}
```

