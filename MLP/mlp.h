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
	using func_type = float(*)(float);

	struct ActFn
	{
		func_type f = nullptr;
		func_type df = nullptr;

		float operator()(float x) const { return f(x); }
		float grad(float x) const { return df(x); }

	};

	extern const ActFn relu_fn;
	extern const ActFn sigmoid_fn;

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
		enum distribution_type { normal, uniform };
		enum initialization_type { He, Xavier };

		mlp() = delete;
		mlp(std::initializer_list<int> layer_sizes, std::initializer_list<ActFn> activation_funcs);
		~mlp() = default;

		void set_values(distribution_type dist_type, initialization_type init_type);
		const Matrix& forward(const Matrix& input);
		void backward(const Matrix& output, const Matrix& target, float learning_rate);

	private:
		std::vector<Matrix> weights;
		std::vector<Matrix> biases;
		std::vector<ActFn> activation_functions;
		std::vector<Matrix> layers;
		size_t num_layers;

	};

	mlp::mlp(std::initializer_list<int> layer_sizes, std::initializer_list<ActFn> activation_funcs)
	{
		if (layer_sizes.size() < 2)
		{
			throw std::invalid_argument("MLP must have at least 2 layers");
		}
		if (activation_funcs.size() != layer_sizes.size() - 1)
		{
			throw std::invalid_argument("Number of activation functions must be equal to number of layers - 1");
		}

		num_layers = layer_sizes.size();

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
		activation_functions = std::vector<ActFn>(activation_funcs);
	}

}
