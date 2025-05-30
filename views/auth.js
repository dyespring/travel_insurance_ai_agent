class AuthService {
    constructor() {
        this.currentUser = null;
    }
    
    async login(firstName, lastName) {
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    first_name: firstName,
                    last_name: lastName
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentUser = data.user;
                localStorage.setItem('currentUser', JSON.stringify(data.user));
                return { success: true, user: data.user };
            } else {
                return { success: false, message: data.message };
            }
        } catch (error) {
            console.error("Login failed:", error);
            return { success: false, message: "Connection error" };
        }
    }
    
    logout() {
        localStorage.removeItem('currentUser');
        this.currentUser = null;
    }
    
    loadSession() {
        const user = JSON.parse(localStorage.getItem('currentUser'));
        if (user) this.currentUser = user;
        return user;
    }
}

// Initialize auth service
const auth = new AuthService();

// Login form handler
document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const firstName = document.getElementById('firstName').value;
    const lastName = document.getElementById('lastName').value;
    
    const result = await auth.login(firstName, lastName);
    
    if (result.success) {
        // Update UI
        document.getElementById('welcomeMessage').textContent = 
            `Welcome back, ${result.user.name.split(' ')[0]}!`;
        
        // Show personalized dashboard
        document.getElementById('loginContainer').classList.add('d-none');
        document.getElementById('chatContainer').classList.remove('d-none');
        
        // Load user-specific data
        loadUserProfile(result.user.user_id);
    } else {
        alert(result.message);
    }
});