# **BERT Service**
Built using `bert-as-service` - [Link](https://bert-as-service.readthedocs.io/en/latest/section/get-start.html)

### Google Bert Models
(Bert Models)[https://github.com/google-research/bert]

## Setup
1. Download BERT model from Google Research Repo
2. Unzip model into `server` directory
3. Reference in `entrypoint.sh` for `-model_dir`
4. Run `docker-compose up`