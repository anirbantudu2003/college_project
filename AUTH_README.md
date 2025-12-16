# User Authentication and Authorization Guide

## Overview
This application now includes a complete user authentication system using Spring Security with form-based login.

## Features Implemented

### 1. User Entity
- Location: `src/main/java/com/example/demo/entity/User.java`
- Fields:
  - `id` (Primary Key)
  - `username` (Unique, required)
  - `email` (Unique, required)
  - `password` (Encrypted with BCrypt)
  - `fullName`
  - `role` (ROLE_USER or ROLE_ADMIN)
  - `enabled` (Account status)
  - `createdAt` and `updatedAt` timestamps

### 2. DTOs (Data Transfer Objects)
- **UserRegistrationDto**: For user registration with validation
  - Fields: username, email, password, confirmPassword, fullName
  - Validation: Min/Max length, Email format, Required fields
  
- **UserLoginDto**: For login requests
  - Fields: username, password, rememberMe

### 3. Repository
- **UserRepository**: JPA repository for User entity
  - Methods:
    - `findByUsername(String username)`
    - `findByEmail(String email)`
    - `existsByUsername(String username)`
    - `existsByEmail(String email)`

### 4. Services

#### UserService
- `registerNewUser(UserRegistrationDto)`: Register new users with validation
- `findByUsername(String)`: Find user by username
- `findByEmail(String)`: Find user by email
- Password encryption using BCrypt

#### CustomUserDetailsService
- Implements Spring Security's `UserDetailsService`
- Loads user details for authentication
- Manages user authorities and roles

### 5. Security Configuration
- Location: `src/main/java/com/example/demo/config/SecurityConfig.java`
- Features:
  - Form-based login (no JWT)
  - BCrypt password encoding
  - Session management
  - Remember-me functionality
  - CSRF protection
  - Role-based access control

### 6. Endpoints Access Control

#### Public Endpoints (No Authentication Required)
- `/register` - User registration page
- `/login` - Login page
- `/css/**`, `/js/**`, `/images/**` - Static resources

#### Authenticated Endpoints (Login Required)
- `/` - Home page
- `/dashboard` - User dashboard
- `/api/predict/**` - Prediction endpoints

#### Admin Only Endpoints
- `/api/model/**` - Model management endpoints

### 7. Controllers

#### AuthController
- `GET /register`: Show registration form
- `POST /register`: Process registration
- `GET /login`: Show login form

#### HomeController
- `GET /`: Home page (shows user info if logged in)
- `GET /dashboard`: User dashboard

### 8. Templates (Thymeleaf)
- `login.html`: Login page with error handling
- `register.html`: Registration form with validation
- `home.html`: Landing page with conditional content
- `dashboard.html`: User dashboard with role-based features
- `error/403.html`: Access denied page

## How to Use

### 1. Start the Application
```bash
mvn spring-boot:run
```

### 2. Access the Application
- Navigate to: `http://localhost:8080`
- You'll be redirected to login page if not authenticated

### 3. Register a New Account
1. Click "Register" or go to `/register`
2. Fill in the form:
   - Full Name (2-100 characters)
   - Username (3-50 characters, unique)
   - Email (valid email format, unique)
   - Password (minimum 6 characters)
   - Confirm Password
3. Click "Create Account"
4. You'll be redirected to login page

### 4. Login
1. Go to `/login`
2. Enter username and password
3. Optionally check "Remember me" for persistent login
4. Click "Login"
5. You'll be redirected to home page

### 5. Logout
- Click "Logout" button in the navbar
- Session will be terminated
- You'll be redirected to login page

## Default User Roles

### ROLE_USER (Default for all registered users)
- Can access:
  - Home page
  - Dashboard
  - Prediction endpoints
  - Profile settings

### ROLE_ADMIN (Must be set manually in database)
- All ROLE_USER permissions plus:
  - Model management
  - Administrative features

## Changing User Role to Admin

To make a user an admin, update the database:
```sql
UPDATE users SET role = 'ROLE_ADMIN' WHERE username = 'your_username';
```

## Security Features

1. **Password Encryption**: All passwords are encrypted using BCrypt
2. **CSRF Protection**: Enabled for form submissions
3. **Session Management**: One active session per user
4. **Remember Me**: Optional persistent login (24 hours)
5. **Account Status**: Can enable/disable user accounts
6. **Role-Based Access**: Different permissions for USER and ADMIN roles

## Database Schema

The `users` table will be automatically created with the following structure:
```sql
CREATE TABLE users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'ROLE_USER',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP
);
```

## API Endpoints Summary

| Endpoint | Method | Access | Description |
|----------|--------|--------|-------------|
| `/` | GET | Public/Authenticated | Home page |
| `/register` | GET/POST | Public | User registration |
| `/login` | GET/POST | Public | User login |
| `/logout` | POST | Authenticated | User logout |
| `/dashboard` | GET | Authenticated | User dashboard |
| `/api/predict/**` | ALL | Authenticated | Prediction API |
| `/api/model/**` | ALL | Admin Only | Model management |

## Troubleshooting

### Issue: Can't login after registration
- **Solution**: Check if the database connection is working
- Verify the user was created: `SELECT * FROM users WHERE username = 'your_username';`

### Issue: "Access Denied" error
- **Solution**: Check user role in database
- Ensure the endpoint allows your role

### Issue: Session expires quickly
- **Solution**: Check "Remember me" option during login
- Or modify `tokenValiditySeconds` in SecurityConfig

### Issue: Registration fails
- **Solution**: 
  - Check if username/email already exists
  - Verify all validation requirements are met
  - Check database connection

## Next Steps

1. **Add Email Verification**: Implement email verification for new users
2. **Password Reset**: Add forgot password functionality
3. **Profile Management**: Allow users to update their profile
4. **User Management**: Create admin panel to manage users
5. **Audit Logging**: Track user activities
6. **Two-Factor Authentication**: Add 2FA for enhanced security

## Testing

To test the authentication system:

1. Register a new user
2. Login with the credentials
3. Access protected endpoints
4. Try accessing admin endpoints (should be denied)
5. Logout and verify session is terminated
6. Test "Remember me" functionality

## Important Notes

- **Never commit** database credentials to version control
- Store sensitive configuration in environment variables
- Regularly update Spring Security dependencies
- Use HTTPS in production
- Implement rate limiting for login attempts
- Add account lockout after failed login attempts
