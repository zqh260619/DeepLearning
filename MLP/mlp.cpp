#include"pch.h"
#include"mlp.h"

namespace MLP
{
	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}
	inline float D_sigmoid(float x)
	{
		return sigmoid(x) * (1.0f - sigmoid(x));
	}
	ActFn sigmoid_fn{ sigmoid, D_sigmoid };

	inline float relu(float x)
	{
		return std::max(0.0f, x);
	}
	inline float D_relu(float x)
	{
		return x > 0.0f ? 1.0f : 0.0f;
	}
	ActFn relu_fn{ relu, D_relu };

}