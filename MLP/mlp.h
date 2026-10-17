#pragma once

#include<iostream>
#include<random>
#include<vector>
#include<initializer_list>

#include"../LinearAlgebra/matrix.h"

namespace MLP
{
	using LinearAlgebra::Matrix;

	class layer
	{
	public:
		layer() = delete;
		~layer() = default;

	private:

	};

	template<typename T>
	class mlp
	{
	public:
		using func_type = T(*)(T);

		mlp() = delete;
		~mlp() = default;

	private:

	};

}
