#pragma once

#include<iostream>
#include<random>
#include<vector>
#include<initializer_list>
#include<cmath>
#include<algorithm>

#include"../LinearAlgebra/matrix.h"

namespace MLP
{
	using LinearAlgebra::Matrix;

	struct ActFn
	{
		using func_type = float(*)(float);

		func_type f = nullptr;
		func_type df = nullptr;

		float operator()(float x) const { return f(x); }
		float grad(float x) const { return df(x); }

	};

	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}
	inline float D_sigmoid(float x)
	{
		return sigmoid(x) * (1.0f - sigmoid(x));
	}

	inline float relu(float x)
	{
		return std::max(0.0f, x);
	}
	inline float D_relu(float x)
	{
		return x > 0.0f ? 1.0f : 0.0f;
	}



	//class layer
	//{
	//public:
	//	layer() = delete;
	//	~layer() = default;

	//private:

	//};

	class mlp
	{
	public:
		using func_type = float(*)(float);

		mlp() = delete;
		mlp(std::initializer_list<int> layer_sizes, std::initializer_list<func_type> activation_funcs);
		~mlp() = default;

	private:
		std::vector<Matrix> weights;
		std::vector<Matrix> biases;
		std::vector<func_type> activation_functions;
		std::vector<Matrix> layers;

	};

	inline mlp::mlp(std::initializer_list<int> layer_sizes, std::initializer_list<func_type> activation_funcs)
	{
		if (layer_sizes.size() < 2)
		{
			throw std::invalid_argument("MLP must have at least 2 layers");
		}
		if (activation_funcs.size() != layer_sizes.size() - 1)
		{
			throw std::invalid_argument("Number of activation functions must be equal to number of layers - 1");
		}

		for(std::size_t i = 0; i < layer_sizes.size() - 1; ++i)
		{
			if (layer_sizes.begin()[i] <= 0 || layer_sizes.begin()[i + 1] <= 0)
			{
				throw std::invalid_argument("Layer sizes must be positive integers");
			}

			weights.push_back(Matrix(layer_sizes.begin()[i + 1], layer_sizes.begin()[i]));
			biases.push_back(Matrix(layer_sizes.begin()[i + 1], 1));
			layers.push_back(Matrix(layer_sizes.begin()[i], 1));
		}

		layers.push_back(Matrix(layer_sizes.begin()[layer_sizes.size() - 1], 1));
		activation_functions = std::vector<func_type>(activation_funcs);
	}

}
