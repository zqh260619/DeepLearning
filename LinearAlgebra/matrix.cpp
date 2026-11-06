#include"pch.h"
#include"matrix.h"

namespace LinearAlgebra
{
	Matrix Matrix::operator*(const Matrix& other) const
	{
		if (this->cols_ != other.rows_)
		{
			throw std::invalid_argument("Incompatible matrix dimensions");
		}

		Matrix result(this->rows_, other.cols_);

		for (size_t i = 0; i < this->rows_; ++i)
		{
			for (size_t j = 0; j < other.cols_; ++j)
			{
				float sum = 0.0f;
				for (size_t k = 0; k < this->cols_; ++k)
				{
					sum += (*this)(i, k) * other(k, j);
				}
				result(i, j) = sum;
			}
		}
		return result;
	}
}