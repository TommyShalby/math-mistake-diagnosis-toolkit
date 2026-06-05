# Security Notes

Do not commit secrets or private study materials.

Never upload:

- `.env`
- API keys
- Google service-account JSON files
- private PDFs
- generated reports containing personal notes
- checkpoint/debug JSON files
- local absolute paths such as `D:\...`

If an API key was ever committed to a public repository, treat it as leaked:

1. delete or rotate the key in the cloud console,
2. remove the secret from the repository,
3. clean the Git history if necessary,
4. check billing and usage logs.
