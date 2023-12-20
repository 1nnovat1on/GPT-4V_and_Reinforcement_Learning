# Vision-Enhanced Reinforcement Learning Agent

This project integrates GPT-4 with Vision (GPT-4V) capabilities into a reinforcement learning environment using Pygame and TensorFlow. The agent in this environment learns to navigate and interact based on both visual and textual inputs, combining traditional reinforcement learning techniques with the cutting-edge ability to process and understand images.

## Features

- Integration with GPT-4 with Vision for enhanced perception.
- A reinforcement learning model built with TensorFlow and Keras.
- Real-time environment simulation using Pygame.
- Asynchronous data fetching for efficient performance.

## Getting Started

### Dependencies

Ensure you have Python 3.x installed on your machine. This project depends on several Python libraries, including:

- Pygame
- NumPy
- TensorFlow
- Keras
- Requests
- Pillow
- aiohttp

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/1nnovat1on/GPT-4V_and_Reinforcement_Learning.git  
cd GPT-4V_and_Reinforcement_Learning
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

### Environment Variables

Set the `OPENAI_API_KEY` environment variable to use the GPT-4 with Vision API:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

### Executing the Program

Run the main script:

```bash
python main.py
```

## Usage

The program simulates an environment where an agent learns to navigate and respond to visual and textual cues. The agent's behavior is influenced by the rewards and penalties defined in the reinforcement learning model.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Authors

Colin Jackson    
Contact: [colinjackson97@icloud.com](mailto:colinjackson97@icloud.com)  
GitHub: [1nnovat1on](https://github.com/1nnovat1on)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- The OpenAI team for GPT-4 with Vision technology.
- Contributors to the TensorFlow, Keras, and Pygame libraries.
