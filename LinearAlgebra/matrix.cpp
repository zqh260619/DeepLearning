#include"pch.h"
#include"matrix.h"

namespace LinearAlgebra
{
	template<typename T>
	Matrix<T> Matrix<T>::operator*(const Matrix<T>& other)
	{
		if (this->cols_ != other.rows_)
		{
			throw std::invalid_argument("Incompatible matrix dimensions");
		}

		Matrix<T> result(this->rows_, other.cols_);

		for (size_t i = 0; i < this->rows_; ++i)
		{
			for (size_t j = 0; j < other.cols_; ++j)
			{
				T sum = T();
				for (size_t k = 0; k < this->cols_; ++k)
				{
					sum += this->data_[*this(i, k)] * other.data_[other(k, j)];
				}
				result.data_[result(i, j)] = sum;
			}
		}
		return result;
	}
}