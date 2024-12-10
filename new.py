import mysql.connector


def get_list_student(subject_id):
    # Kết nối đến cơ sở dữ liệu MySQL
    mydb = mysql.connector.connect(
        host="bug0usy6grg9homkxqes-mysql.services.clever-cloud.com",
        user="uu42yrnrehb2xqsx",
        password="99v5Iw2SYzHpHUiJG3Tr",
        database="bug0usy6grg9homkxqes"
    )

    cursor = None  # Khởi tạo biến cursor với giá trị None

    try:
        # Tạo con trỏ cơ sở dữ liệu
        cursor = mydb.cursor(dictionary=True)  # Sử dụng dictionary=True để trả về kết quả dưới dạng từ điển

        # Truy vấn danh sách sinh viên từ bảng trung gian
        query = """
        SELECT sv.MSSV 
        FROM sinhvien_hocphan shp
        JOIN sinh_vien sv ON shp.ma_sinh_vien = sv.MSSV
        WHERE shp.ma_hoc_phan = %s
        """
        cursor.execute(query, (subject_id,))

        # Lấy kết quả
        students = cursor.fetchall()
        mssv_list = [student['MSSV'] for student in students]
        # Nếu không có sinh viên nào được tìm thấy
        if not mssv_list:
            return []  # Trả về danh sách rỗng nếu không có sinh viên nào

        # Trả về danh sách sinh viên dưới dạng danh sách Python
        return mssv_list

    except Exception as e:
        print(f"Lỗi khi lấy danh sách dữ liệu: {e}")
        return []  # Trả về danh sách rỗng nếu có lỗi xảy ra

    finally:
        if cursor:
            cursor.close()  # Đảm bảo rằng con trỏ được đóng sau khi sử dụng
        if mydb.is_connected():
            mydb.close()  # Đóng kết nối cơ sở dữ liệu


# Ví dụ về cách sử dụng hàm để lấy danh sách sinh viên
subject_id = '2'  # Giá trị mẫu của subject_id
students_list = get_list_student(subject_id)
print(students_list)

