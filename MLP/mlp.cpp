#include"pch.h"
#include"mlp.h"

namespace MLP
{
	namespace
	{
		float sigmoid(float x)
		{
			return 1.0f / (1.0f + std::exp(-x));
		}
		float D_sigmoid(float x)
		{
			return sigmoid(x) * (1.0f - sigmoid(x));
		}

		float relu(float x)
		{
			return std::max(0.0f, x);
		}
		float D_relu(float x)
		{
			return x > 0.0f ? 1.0f : 0.0f;
		}

	}

	const ActFn sigmoid_fn{ sigmoid, D_sigmoid };
	const ActFn relu_fn{ relu, D_relu };

	void mlp::set_values(distribution_type dist_type, initialization_type init_type)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		if (dist_type == He)
		{
			if (init_type == normal)
			{
				float mu = 0.0f;

				for (std::size_t i = 0; i < weights.size(); i++)
				{
					float sigma = sqrt(2.0f / weights[i].cols());
					weights[i].set_random_normal(mu, sigma, gen);
				}
			}
			else if (init_type == uniform)
			{
				for (std::size_t i = 0; i < weights.size(); i++)
				{
					float limit = sqrt(6.0f / weights[i].cols());
					weights[i].set_random_uniform(-limit, limit, gen);
				}
			}
			else
			{
				throw std::invalid_argument("Invalid initialization type");
			}
		}
		else if (dist_type == Xavier)
		{
			if (init_type == normal)
			{
				float mu = 0.0f;
				for (std::size_t i = 0; i < weights.size(); i++)
				{
					float sigma = sqrt(2.0f / (weights[i].cols() + weights[i].rows()));
					weights[i].set_random_normal(mu, sigma, gen);
				}
			}
			else if (init_type == uniform)
			{
				for (std::size_t i = 0; i < weights.size(); i++)
				{
					float limit = sqrt(6.0f / (weights[i].cols() + weights[i].rows()));
					weights[i].set_random_uniform(-limit, limit, gen);
				}
			}
			else
			{
				throw std::invalid_argument("Invalid initialization type");
			}
		}
		else
		{
			throw std::invalid_argument("Invalid distribution type");
		}
	}

	const Matrix& mlp::forward(const Matrix& input)
	{
		layers[0] = input;
		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			Matrix z = weights[i] * layers[i] + biases[i];
			layers[i + 1] = z.calculate(activation_functions[i].f);
		}
		return layers[num_layers - 1];
	}

}