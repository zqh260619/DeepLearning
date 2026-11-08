#pragma once

#include<vector>
#include<iterator>
#include<stdexcept>
#include<random>

namespace LinearAlgebra
{
	class Matrix
	{
	public:
		using func_type = float(*)(float);

		Matrix() = delete;
		Matrix(size_t rows, size_t cols) :
			rows_(rows), cols_(cols), data_(rows* cols, 0.0f), size_(rows* cols) {}
		Matrix(const Matrix& other) = default;
		Matrix(Matrix&& other) noexcept = default;
		Matrix& operator=(const Matrix& other) = default;
		Matrix& operator=(Matrix&& other) noexcept = default;
		~Matrix() = default;

		float& operator()(size_t row, size_t col) { return data_[get_index(row, col)]; }
		const float& operator()(size_t row, size_t col) const { return data_[get_index(row, col)]; }

		Matrix calculate(func_type func) const;
		void set_random_normal(float mu, float sigma, std::mt19937& gen);
		void set_random_uniform(float min, float max, std::mt19937& gen);

		Matrix operator+(const Matrix& other) const;
		Matrix operator-() const;
		Matrix operator-(const Matrix& other) const { return *this + (-other); }
		Matrix operator*(float scalar) const;
		Matrix operator*(const Matrix& other) const;
		Matrix& operator+=(const Matrix& other) { *this = *this + other; return *this; }
		Matrix& operator-=(const Matrix& other) { *this = *this - other; return *this; }
		Matrix& operator*=(float scalar) { *this = *this * scalar; return *this; }
		Matrix& operator*=(const Matrix& other) { *this = *this * other; return *this; }

	private:
		size_t rows_, cols_;
		size_t size_;//total size of the matrix
		std::vector<float> data_;

		size_t get_index(size_t row, size_t col) const
		{
			if (row >= rows_ || col >= cols_)
			{
				throw std::out_of_range("Matrix index out of bounds");
			}
			return row * cols_ + col;
		}

	};

	inline Matrix Matrix::calculate(func_type func) const
	{
		Matrix result(rows_, cols_);

		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = func(this->data_[i]);
		}
		return result;
	}

	inline void Matrix::set_random_normal(float mu, float sigma,std::mt19937& gen)
	{
		std::normal_distribution<float> dist(mu, sigma);

		for (auto& i : data_)
		{
			i = dist(gen);
		}
	}

	inline void Matrix::set_random_uniform(float min, float max, std::mt19937& gen)
	{
		std::uniform_real_distribution<float> dist(min, max);

		for (auto& i : data_)
		{
			i = dist(gen);
		}
	}

	inline Matrix Matrix::operator+(const Matrix& other) const
	{
		if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
		{
			throw std::invalid_argument("Incompatible matrix dimensions");
		}

		Matrix result(rows_, cols_);

		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = this->data_[i] + other.data_[i];
		}
		return result;
	}

	inline Matrix Matrix::operator-() const
	{
		Matrix result(rows_, cols_);
		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = -this->data_[i];
		}
		return result;
	}

	inline Matrix Matrix::operator*(float scalar) const
	{
		Matrix result(rows_, cols_);
		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = this->data_[i] * scalar;
		}
		return result;
	}
}
